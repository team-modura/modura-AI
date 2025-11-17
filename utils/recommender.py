# recommender.py

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from lightfm import LightFM
from lightfm.data import Dataset
from scipy.sparse import csr_matrix
from sqlalchemy.orm import Session
from geopy.distance import geodesic

from models.db_models import Category, ContentCategory, ContentPlace, Place, User, UserContentLikes, UserPlaceLikes, Content, UserCategory
from models.models import LocationEnum
from utils.helpers import calculate_distance, distance_to_score, geocode

class LightFMRecommender:
    """
    LightFM 기반 하이브리드 추천 시스템
    - Collaborative Filtering (사용자-컨텐츠 상호작용)
    - Content-based (장르, 거리 등 메타데이터)
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.model = None
        self.dataset = None
        self.user_features = None
        self.item_features = None
        self.user_id_map = {}
        self.item_id_map = {}
        self.category_cache = {}
        
        # 카테고리 정보 미리 로드
        self._load_categories()

        # 모델 초기화 및 학습
        self._build_model()
    
    def _load_categories(self):
        """카테고리 정보를 메모리에 캐싱"""
        categories = self.db.query(Category).all()
        self.category_cache = {cat.id: cat.name for cat in categories}
        print(f"Loaded {len(self.category_cache)} categories")
    
    def _build_model(self):
        """LightFM 모델 구축 및 학습"""
        # 1. 데이터 로드
        users = self.db.query(User).all()
        contents = self.db.query(Content).all()
        interactions = self.db.query(UserContentLikes).all()
        
        if not users or not contents or not interactions:
            print("Warning: Insufficient data for training")
            return
        
        print(f"Building model with {len(users)} users, {len(contents)} contents, {len(interactions)} interactions")
        
        # 2. Dataset 생성
        self.dataset = Dataset()
        
        user_ids = [u.id for u in users]
        item_ids = [c.id for c in contents]
        
        # 3. 사용자 특성 수집
        user_features_dict = {}
        for user in users:
            features = []
            
            # 지역 특성
            if user.address:
                safe_address = user.address.replace(" ", "_")
                features.append(f"location:{safe_address}")
            
            # 선호 카테고리 (UserCategory 테이블에서 조회)
            user_cats = self.db.query(UserCategory).filter(
                UserCategory.user_id == user.id
            ).all()
            
            for uc in user_cats:
                cat_name = self.category_cache.get(uc.category_id, f"cat_{uc.category_id}")
                features.append(f"user_pref:{cat_name}")
            
            user_features_dict[user.id] = features
        
        # 4. 아이템 특성 수집
        item_features_dict = {}
        for content in contents:
            features = []
            
            # 컨텐츠 타입
            features.append(f"type:{content.type}")
            
            # 컨텐츠 카테고리 (ContentCategory 테이블에서 조회)
            content_cats = self.db.query(ContentCategory).filter(
                ContentCategory.content_id == content.id
            ).all()
            
            for cc in content_cats:
                cat_name = self.category_cache.get(cc.category_id, f"cat_{cc.category_id}")
                features.append(f"genre:{cat_name}")
            
            # 제작 연도 구간 (10년 단위)
            if content.year:
                decade = (content.year // 10) * 10
                features.append(f"decade:{decade}s")
            
            item_features_dict[content.id] = features
        
        # 5. 모든 특성 수집
        all_user_features = set()
        for features in user_features_dict.values():
            all_user_features.update(features)
        
        all_item_features = set()
        for features in item_features_dict.values():
            all_item_features.update(features)
        
        print(f"User features: {len(all_user_features)}, Item features: {len(all_item_features)}")
        
        # 6. Dataset fit
        self.dataset.fit(
            users=user_ids,
            items=item_ids,
            user_features=all_user_features,
            item_features=all_item_features
        )
        
        # 7. 상호작용 행렬 생성 (찜한 컨텐츠 = 긍정적 피드백)
        interactions_data = [(i.user_id, i.content_id) for i in interactions]
        interaction_matrix, _ = self.dataset.build_interactions(interactions_data)
        
        # 8. 특성 행렬 생성
        user_features_data = [
            (user_id, features)
            for user_id, features in user_features_dict.items()
        ]
        self.user_features = self.dataset.build_user_features(user_features_data)
        
        item_features_data = [
            (item_id, features)
            for item_id, features in item_features_dict.items()
        ]
        self.item_features = self.dataset.build_item_features(item_features_data)
        
        # 9. LightFM 모델 학습
        self.model = LightFM(
            no_components=30,      # 잠재 요인 차원
            loss='warp',           # WARP loss (랭킹 최적화)
            learning_rate=0.05,
            item_alpha=0.0,        # 아이템 정규화
            user_alpha=0.0,        # 사용자 정규화
            random_state=42
        )
        
        print("Training LightFM model...")
        self.model.fit(
            interactions=interaction_matrix,
            user_features=self.user_features,
            item_features=self.item_features,
            epochs=20,
            num_threads=4,
            verbose=True
        )
        
        # ID 매핑 저장
        self.user_id_map = self.dataset.mapping()[0]
        self.item_id_map = self.dataset.mapping()[2]
        
        print(f"Model training completed!")
    
    def predict_for_user(self, user_id: int, n_items: int = 100) -> List[Tuple[int, float]]:
        """
        특정 사용자에 대한 추천 점수 예측 (CF 점수)
        Returns: [(content_id, score), ...]
        """
        if not self.model or user_id not in self.user_id_map:
            return []
        
        # 내부 사용자 ID로 변환
        internal_user_id = self.user_id_map[user_id]
        
        # 모든 아이템에 대한 점수 예측
        item_ids = list(self.item_id_map.keys())
        internal_item_ids = [self.item_id_map[iid] for iid in item_ids]
        
        scores = self.model.predict(
            user_ids=internal_user_id,
            item_ids=internal_item_ids,
            user_features=self.user_features,
            item_features=self.item_features
        )
        
        # 이미 좋아요한 아이템 제외
        liked_items = self.db.query(UserContentLikes.content_id)\
            .filter(UserContentLikes.user_id == user_id).all()
        liked_item_ids = {item[0] for item in liked_items}
        
        # (content_id, score) 튜플 생성 및 정렬
        recommendations = [
            (item_ids[i], float(scores[i])) 
            for i in range(len(item_ids))
            if item_ids[i] not in liked_item_ids
        ]
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_items]
    
    def calculate_genre_score(self, user_id: int, content_id: int) -> float:
        """
        장르 매칭 점수 계산 (Jaccard Similarity)
        사용자 선호 장르 vs 컨텐츠 장르
        """
        # 사용자 선호 장르
        user_categories = self.db.query(UserCategory.category_id)\
            .filter(UserCategory.user_id == user_id).all()
        user_cat_set = {cat[0] for cat in user_categories}
        
        if not user_cat_set:
            return 0.0
        
        # 컨텐츠 장르
        content_categories = self.db.query(ContentCategory.category_id)\
            .filter(ContentCategory.content_id == content_id).all()
        content_cat_set = {cat[0] for cat in content_categories}
        
        if not content_cat_set:
            return 0.0
        
        # Jaccard Similarity: 교집합 / 합집합
        intersection = len(user_cat_set & content_cat_set)
        union = len(user_cat_set | content_cat_set)
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_distance_score(self, user_id: int, content_id: int) -> float:
        """
        사용자 위치와 컨텐츠 촬영지 간 거리 점수
        거리가 가까울수록 높은 점수
        """
        # 사용자 주소
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user or not user.address:
            return 0.0
        
        user_coords = geocode(user.address)
        if not user_coords:
            return 0.0
        
        # 컨텐츠와 연결된 촬영지들
        content_places = self.db.query(ContentPlace, Place)\
            .join(Place, ContentPlace.place_id == Place.id)\
            .filter(ContentPlace.content_id == content_id)\
            .all()
        
        if not content_places:
            return 0.0
        
        # 가장 가까운 촬영지까지의 거리
        min_distance = float('inf')
        for cp, place in content_places:
            if place.latitude and place.longitude:
                place_coords = (place.latitude, place.longitude)
                distance = calculate_distance(user_coords, place_coords)
                min_distance = min(min_distance, distance)
        
        if min_distance == float('inf'):
            return 0.0
        
        # 거리를 점수로 변환 (100km 이내는 선형, 그 이상은 0)
        return distance_to_score(min_distance, max_distance=100.0)
    
    def get_hybrid_recommendations(
        self, 
        user_id: int, 
        n_recommendations: int = 10,
        cf_weight: float = 0.6,
        genre_weight: float = 0.25,
        distance_weight: float = 0.15
    ) -> List[Dict]:
        """
        하이브리드 추천: CF + 장르 + 거리
        """
        # 1. CF 점수 (LightFM 예측)
        cf_recommendations = self.predict_for_user(user_id, n_items=50)
        
        if not cf_recommendations:
            print(f"No CF recommendations for user {user_id}")
            return []
        
        # 2. 각 아이템에 대해 종합 점수 계산
        final_scores = []
        
        # CF 점수 정규화를 위한 min/max
        cf_scores = [score for _, score in cf_recommendations]
        cf_min, cf_max = min(cf_scores), max(cf_scores)
        
        for content_id, cf_score in cf_recommendations:
            # 정규화된 CF 점수 (0~1)
            normalized_cf = (cf_score - cf_min) / (cf_max - cf_min + 1e-10)
            
            # 장르 점수
            genre_score = self.calculate_genre_score(user_id, content_id)
            
            # 거리 점수
            distance_score = self.calculate_distance_score(user_id, content_id)
            
            # 가중 평균
            final_score = (
                cf_weight * normalized_cf +
                genre_weight * genre_score +
                distance_weight * distance_score
            )
            
            final_scores.append({
                'content_id': content_id,
                'score': final_score,
                'cf_score': float(cf_score),
                'genre_score': genre_score,
                'distance_score': distance_score
            })
        
        # 최종 점수로 정렬
        final_scores.sort(key=lambda x: x['score'], reverse=True)
        
        return final_scores[:n_recommendations]
    
    def get_content_details(self, content_id: int) -> Dict:
        """컨텐츠 상세 정보 조회 (디버깅용)"""
        content = self.db.query(Content).filter(Content.id == content_id).first()
        if not content:
            return {}
        
        # 장르 정보
        categories = self.db.query(ContentCategory, Category)\
            .join(Category, ContentCategory.category_id == Category.id)\
            .filter(ContentCategory.content_id == content_id)\
            .all()
        
        genre_names = [cat.name for _, cat in categories]
        
        return {
            'id': content.id,
            'title_kr': content.title_kr,
            'title_eng': content.title_eng,
            'year': content.year,
            'genres': genre_names,
            'type': content.type
        }