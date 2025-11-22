# recommender.py

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from lightfm import LightFM
from lightfm.data import Dataset
from scipy.sparse import csr_matrix
from sqlalchemy.orm import Session

from models.db_models import (
    Category,
    ContentCategory,
    ContentPlace,
    Place,
    User,
    UserContentLikes,
    Content,
    UserCategory
)
from models.models import LocationEnum
from utils.helpers import calculate_distance, distance_to_score, geocode


class LightFMRecommender:
    """
    LightFM 기반 하이브리드 추천 시스템
    - 1차 후보 생성: CF / 장르 / 거리 별로 각각 후보 추출
    - 2차 랭킹: CF + 장르 + 거리 점수로 최종 랭킹 산출
    """

    def __init__(self, db: Session):
        self.db = db
        self.model = None
        self.dataset = None
        self.user_features = None
        self.item_features = None
        self.user_id_map: Dict[int, int] = {}
        self.item_id_map: Dict[int, int] = {}
        self.category_cache: Dict[int, str] = {}

        self._load_categories()
        self._build_model()

    def _load_categories(self):
        """카테고리 정보를 메모리에 캐싱"""
        categories = self.db.query(Category).all()
        self.category_cache = {cat.id: cat.name for cat in categories}
        
    def _build_model(self):
        """LightFM 모델 구축 및 학습"""
        # 1. 데이터 로드
        users = self.db.query(User).all()
        contents = self.db.query(Content).all()
        interactions = self.db.query(UserContentLikes).all()

        if not users or not contents or not interactions:
            return

        # 2. Dataset 생성
        self.dataset = Dataset()

        user_ids = [u.id for u in users]
        item_ids = [c.id for c in contents]

        # 3. 사용자 특성 수집
        user_features_dict: Dict[int, List[str]] = {}
        for user in users:
            features: List[str] = []

            # 지역 특성
            if user.address:
                safe_address = user.address.replace(" ", "_")
                features.append(f"location:{safe_address}")

            # 선호 카테고리 (UserCategory 테이블에서 조회)
            user_cats = (
                self.db.query(UserCategory)
                .filter(UserCategory.user_id == user.id)
                .all()
            )

            for uc in user_cats:
                cat_name = self.category_cache.get(uc.category_id, f"cat_{uc.category_id}")
                features.append(f"user_pref:{cat_name}")

            user_features_dict[user.id] = features

        # 4. 아이템 특성 수집
        item_features_dict: Dict[int, List[str]] = {}
        for content in contents:
            features: List[str] = []

            # 컨텐츠 카테고리
            content_cats = (
                self.db.query(ContentCategory)
                .filter(ContentCategory.content_id == content.id)
                .all()
            )

            for cc in content_cats:
                cat_name = self.category_cache.get(cc.category_id, f"cat_{cc.category_id}")
                features.append(f"genre:{cat_name}")

            item_features_dict[content.id] = features

        # 5. 모든 특성 수집
        all_user_features = set()
        for features in user_features_dict.values():
            all_user_features.update(features)

        all_item_features = set()
        for features in item_features_dict.values():
            all_item_features.update(features)

        # 6. Dataset fit
        self.dataset.fit(
            users=user_ids,
            items=item_ids,
            user_features=all_user_features,
            item_features=all_item_features,
        )

        # 7. 상호작용 행렬 생성
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
            no_components=30,
            loss="warp",  # 랭킹 최적화
            learning_rate=0.05,
            item_alpha=0.0,
            user_alpha=0.0,
            random_state=42,
        )

        self.model.fit(
            interactions=interaction_matrix,
            user_features=self.user_features,
            item_features=self.item_features,
            epochs=20,
            num_threads=4,
            verbose=True,
        )

        # ID 매핑 저장
        self.user_id_map = self.dataset.mapping()[0]
        self.item_id_map = self.dataset.mapping()[2]

    def predict_for_user(self, user_id: int, n_items: int = 100) -> List[Tuple[int, float]]:
        """
        특정 사용자에 대한 CF 기반 추천 점수 예측
        Returns: [(content_id, cf_score), ...] (cf_score는 raw score)
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
            item_features=self.item_features,
        )

        # 이미 좋아요한 아이템 제외
        liked_items = (
            self.db.query(UserContentLikes.content_id)
            .filter(UserContentLikes.user_id == user_id)
            .all()
        )
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
        장르 매칭 점수 계산
        사용자 선호 장르 vs 컨텐츠 장르
        """
        user_categories = (
            self.db.query(UserCategory.category_id)
            .filter(UserCategory.user_id == user_id)
            .all()
        )
        user_cat_set = {cat[0] for cat in user_categories}

        content_categories = (
            self.db.query(ContentCategory.category_id)
            .filter(ContentCategory.content_id == content_id)
            .all()
        )
        content_cat_set = {cat[0] for cat in content_categories}

        if not user_cat_set or not content_cat_set:
          return 0.0
        
        intersection = len(user_cat_set & content_cat_set)
        score = intersection / len(user_cat_set)

        return score
    
    def calculate_distance_score(self, user_id: int, content_id: int) -> float:
        """
        사용자 위치와 컨텐츠 촬영지 간 거리 점수
        (구체적인 변환 로직은 utils.helpers.distance_to_score 에 위임)
        """
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user or not user.address:
            return 0.0

        user_coords = geocode(user.address)
        if not user_coords:
            return 0.0

        content_places = (
            self.db.query(ContentPlace, Place)
            .join(Place, ContentPlace.place_id == Place.id)
            .filter(ContentPlace.content_id == content_id)
            .all()
        )

        if not content_places:
            return 0.0

        min_distance = float("inf")
        for cp, place in content_places:
            if place.latitude is not None and place.longitude is not None:
                place_coords = (place.latitude, place.longitude)
                distance = calculate_distance(user_coords, place_coords)
                min_distance = min(min_distance, distance)

        if min_distance == float("inf"):
            return 0.0
        return distance_to_score(min_distance)

    def get_genre_candidates(self, user_id: int, top_k: int = 200) -> List[int]:
        """
        장르 기반 후보군 (user 선호 장르와 겹치는 컨텐츠들 중 상위 top_k)
        """
        user_categories = (
            self.db.query(UserCategory.category_id)
            .filter(UserCategory.user_id == user_id)
            .all()
        )
        user_cat_set = {cat[0] for cat in user_categories}
        if not user_cat_set:
            return []

        # 컨텐츠별 장르 미리 로딩
        content_cats = self.db.query(ContentCategory).all()
        content_cat_map: Dict[int, set] = {}
        for cc in content_cats:
            content_cat_map.setdefault(cc.content_id, set()).add(cc.category_id)

        scored: List[Tuple[int, int]] = []
        for content_id, cat_set in content_cat_map.items():
            intersection = len(user_cat_set & cat_set)
            if intersection > 0:
                scored.append((content_id, intersection))

        #내림차순 정렬
        scored.sort(key=lambda x: x[1], reverse=True)

        return [cid for cid, _ in scored[:top_k]]

    def get_distance_candidates(self, user_id: int, top_k: int = 200) -> List[int]:
        """
        거리 기반 후보군 (사용자와 가까운 촬영지의 컨텐츠 상위 top_k)
        """
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user or not user.address:
            return []

        user_coords = geocode(user.address)
        if not user_coords:
            return []

        # 모든 컨텐츠-장소를 가져와서 컨텐츠별 최소 거리 계산
        rows = (
            self.db.query(ContentPlace.content_id, Place.latitude, Place.longitude)
            .join(Place, ContentPlace.place_id == Place.id)
            .all()
        )

        if not rows:
            return []

        # content_id 별 최소 거리 계산
        min_dist_by_content: Dict[int, float] = {}
        for content_id, lat, lon in rows:
            if lat is None or lon is None:
                continue
            place_coords = (lat, lon)
            d = calculate_distance(user_coords, place_coords)
            if content_id not in min_dist_by_content:
                min_dist_by_content[content_id] = d
            else:
                min_dist_by_content[content_id] = min(min_dist_by_content[content_id], d)

        if not min_dist_by_content:
            return []

        # 거리 → 점수 변환 후 정렬
        scored: List[Tuple[int, float]] = []
        for cid, dist in min_dist_by_content.items():
            score = distance_to_score(dist)
            scored.append((cid, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in scored[:top_k]]

 

    def get_hybrid_recommendations(
        self,
        user_id: int,
        n_recommendations: int = 10,
        cf_weight: float = 0.25,
        genre_weight: float = 0.60,
        distance_weight: float = 0.15,
        cf_top_k: int = 200,
        genre_top_k: int = 200,
        distance_top_k: int = 200,
    ) -> List[Dict]:
        """
        1차 필터링:
          - CF 후보: LightFMTopK
          - 장르 후보: User-Content 장르 매칭 기반 TopK
          - 거리 후보: 사용자와 가까운 컨텐츠 TopK

        2차 랭킹:
          - 위 세 후보군 합치기 (중복 제거)
          - 각 컨텐츠별로 CF / 장르 / 거리 점수 계산
          - 가중합으로 최종 점수 산정 후 상위 N개 반환
        """
        # 1. CF 후보
        cf_recommendations = self.predict_for_user(user_id, n_items=cf_top_k)
        cf_score_raw: Dict[int, float] = {cid: score for cid, score in cf_recommendations}

        if not cf_recommendations:
            return []

        genre_candidates = self.get_genre_candidates(user_id, top_k=genre_top_k)

        distance_candidates = self.get_distance_candidates(user_id, top_k=distance_top_k)

        # 2. 후보군 합치기
        candidate_ids = set(cf_score_raw.keys()) | set(genre_candidates) | set(distance_candidates)

        if not candidate_ids:
            return []

        # CF 점수 정규화 (0~1)
        cf_scores_list = list(cf_score_raw.values())
        cf_min, cf_max = min(cf_scores_list), max(cf_scores_list)
        cf_range = cf_max - cf_min if cf_max != cf_min else 1.0

        def get_normalized_cf(cid: int) -> float:
            if cid not in cf_score_raw:
                return 0.0
            return (cf_score_raw[cid] - cf_min) / (cf_range + 1e-10)

        final_scores: List[Dict] = []

        for content_id in candidate_ids:
            normalized_cf = get_normalized_cf(content_id)
            genre_score = self.calculate_genre_score(user_id, content_id)
            distance_score = self.calculate_distance_score(user_id, content_id)

            final_score = (
                cf_weight * normalized_cf
                + genre_weight * genre_score
                + distance_weight * distance_score
            )

            final_scores.append(
                {
                    "content_id": content_id,
                    "score": final_score,
                    "cf_score": float(cf_score_raw.get(content_id, 0.0)),
                    "genre_score": genre_score,
                    "distance_score": distance_score,
                }
            )

        # 최종 점수로 정렬
        final_scores.sort(key=lambda x: x["score"], reverse=True)

        return final_scores[:n_recommendations]

    def get_content_details(self, content_id: int) -> Dict:
        """컨텐츠 상세 정보 조회 (디버깅용)"""
        content = self.db.query(Content).filter(Content.id == content_id).first()
        if not content:
            return {}

        # 장르 정보
        categories = (
            self.db.query(ContentCategory, Category)
            .join(Category, ContentCategory.category_id == Category.id)
            .filter(ContentCategory.content_id == content_id)
            .all()
        )

        genre_names = [cat.name for _, cat in categories]

        return {
            "id": content.id,
            "genres": genre_names
        }
