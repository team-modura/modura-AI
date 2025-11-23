from typing import Optional, List
from fastapi import Depends, FastAPI, HTTPException, Query, status
from utils.helpers import distance_to_score, geocode
from utils.place_recommender import PlaceCFRecommender, batch_distance
from sqlalchemy.orm import Session
from sqlalchemy import func
from utils.database import get_db, engine
from models.db_models import Base, User, Content, UserContentLikes, UserPlaceLikes, Place, ContentPlace, PlaceReviews
from models.models import RecommendationResponse, ContentRecommendation, MapRecommendationResponse, PlaceRecommendation
from utils.recommender import LightFMRecommender
import numpy as np
from utils.s3_service import s3_service

# 테이블 생성 (최초 1회만 실행)
Base.metadata.create_all(bind=engine)

# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI(title="flicker Recommendation API")

# 추천 시스템 인스턴스 초기화
recommender = None
place_recommender = None

@app.on_event("startup")
async def startup_event():
    global recommender, place_recommender
    db = next(get_db())
    try:
        recommender = LightFMRecommender(db)
        place_recommender = PlaceCFRecommender(db)
    except Exception as e:
        print(f"❌ Failed to initialize recommenders: {e}")
        import traceback
        traceback.print_exc()
        recommender = None
        place_recommender = None
    finally:
        db.close()

        
@app.get("/")
def read_root():
    return {"message": "Flicker Recommendation API is running!"}

@app.get("/health")
def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "recommender_ready": recommender is not None and recommender.model is not None,
        "place_recommender_ready": place_recommender is not None and place_recommender.model is not None
    }

@app.get('/recommend/contents/{user_id}', response_model=RecommendationResponse)
def recommend_content(user_id: int, db: Session = Depends(get_db)):
    """
    특정 사용자에게 컨텐츠 추천
    """
    # 사용자 존재 확인
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # 추천 시스템이 초기화되지 않은 경우
    if not recommender:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommender system not initialized"
        )
    
    try:
        # 하이브리드 추천 수행
        recommendations = recommender.get_hybrid_recommendations(
            user_id=user_id,
            n_recommendations=10,
            include_liked_info=True
        )
        
        content_ids = [rec['content_id'] for rec in recommendations]
        contents = db.query(Content).filter(Content.id.in_(content_ids)).all()
        content_map = {c.id: c for c in contents}
        
        # 컨텐츠 정보 조회 및 응답 생성
        result = []
        for rec in recommendations:
            content = content_map.get(rec['content_id'])
            if content:
                result.append(ContentRecommendation(
                    id=content.id,
                    name=content.title_kr,
                    isLiked=rec.get('is_liked', False),
                    thumbnail=content.thumbnail
                ))
        
        return RecommendationResponse(
            user_id=user_id,
            placeList=result,
            total_count=len(result)
        )
    
    except Exception as e:
        db.rollback()
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Recommendation failed: {str(e)}"
        )

@app.get("/recommend/map/{user_id}", response_model=MapRecommendationResponse)
def recommend_by_location(
    user_id: int,
    lat: Optional[float] = Query(None, description="User latitude"),
    lon: Optional[float] = Query(None, description="User longitude"),
    db: Session = Depends(get_db)
):
    """
    지도 기반 추천 (사용자 위치 기반 추천)
    - 유저의 현재 위도/경도 기준으로 가까운 촬영지 컨텐츠 추천
    """
    # 사용자 존재 여부 체크
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    if not place_recommender:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommender system not initialized"
        )

    try:
        if lat is None or lon is None:
            if not user.address:
                raise HTTPException(
                    status_code=400,
                    detail="User location not provided (lat/lon missing, and user has no address)"
                )
            user_coords = geocode(user.address)
            if not user_coords:
                raise HTTPException(
                    status_code=400,
                    detail="Failed to geocode user.address"
                )
        else:
            user_coords = (lat, lon)

        liked_place_rows = (
            db.query(UserPlaceLikes.place_id)
            .filter(UserPlaceLikes.user_id == user_id)
            .all()
        )
        
        liked_place_ids = {row[0] for row in liked_place_rows}

        cf_candidates = place_recommender.get_cf_candidates(
            user_id, 
            top_k=100
        )

        places = db.query(Place).filter(
          Place.latitude.isnot(None),
          Place.longitude.isnot(None)
        ).all()

        if not places:
            return MapRecommendationResponse(placeList=[])

        
        coords_list = [(p.latitude, p.longitude) for p in places]
        coords_array = np.array(coords_list, dtype=float)  # shape (N, 2)

        distances = batch_distance(user_coords, coords_array)  # (N,)

        # 4) 거리 점수 + CF 점수 → 하이브리드
        result: list[tuple[Place, float]] = []
        for idx, place in enumerate(places):
            d = float(distances[idx])
            dist_score = distance_to_score(d)
            cf_score = cf_candidates.get(place.id, 0.0)

            # Hybrid (거리 70%, CF 30%)
            final_score = cf_score * 0.3 + dist_score * 0.7
            result.append((place, final_score))

        # 점수 내림차순 정렬
        result.sort(key=lambda x: x[1], reverse=True)
        top_places = result[:20]

        top_place_ids = [place.id for place, _ in top_places]
        top_liked_ids = liked_place_ids & set(top_place_ids)

        place_contents_rows = (
            db.query(ContentPlace.place_id, Content.title_kr)
            .join(Content, ContentPlace.content_id == Content.id)
            .filter(ContentPlace.place_id.in_(top_place_ids))
            .all()
        )

        place_contents_map = {}
        for place_id, title in place_contents_rows:
            place_contents_map.setdefault(place_id, []).append(title)

        
        rating_rows = (db.query(
                PlaceReviews.place_id,
                func.avg(PlaceReviews.rating).label('avg_rating'),
                func.count(PlaceReviews.id).label('review_count')
            )
            .filter(PlaceReviews.place_id.in_(top_place_ids))
            .group_by(PlaceReviews.place_id)
            .all()
        )
        place_stats = {pid: (avg or 0.0, cnt or 0) for pid, avg, cnt in rating_rows}
        
        place_list: list[PlaceRecommendation] = []
        
        thumbnail_url = None
        if place.thumbnail and isinstance(place.thumbnail, str) and place.thumbnail.strip():
            thumbnail_url = s3_service.generate_view_presigned_url(place.thumbnail)

        for place, _score in top_places:
            avg_rating, review_count = place_stats.get(place.id, (0.0, 0))
            
            place_list.append(
                PlaceRecommendation(
                    id=place.id,
                    name=place.name,
                    isLiked=place.id in top_liked_ids,
                    thumbnail=thumbnail_url,
                    rating=avg_rating,
                    reviewCount=review_count,
                    latitude=place.latitude,
                    longitude=place.longitude,
                    content=place_contents_map.get(place.id, [])
                )
            )

        return MapRecommendationResponse(placeList=place_list)

    except Exception as e:
        db.rollback()
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Map recommendation failed: {str(e)}"
        )


@app.post('/recommend/retrain')
def retrain_model(db: Session = Depends(get_db)):
    """
    추천 모델 재학습 (관리자용 엔드포인트)
    """
    global recommender, place_recommender
    try:
        recommender = LightFMRecommender(db)
        place_recommender = PlaceCFRecommender(db)
        return {"message": "Model retrained successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retraining failed: {str(e)}"
        )
    