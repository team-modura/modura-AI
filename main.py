from fastapi import Depends, FastAPI, HTTPException, status
from sqlalchemy.orm import Session
from utils.database import get_db, engine
from models.db_models import Base, User, Content
from models.models import RecommendationResponse, ContentRecommendation
from utils.recommender import LightFMRecommender


# 테이블 생성 (최초 1회만 실행)
Base.metadata.create_all(bind=engine)

# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI(title="flicker Recommendation API")

# 추천 시스템 인스턴스 초기화
recommender = None

@app.on_event("startup")
async def startup_event():
    global recommender
    db = next(get_db())
    try:
        recommender = LightFMRecommender(db)
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
        "recommender_ready": recommender is not None and recommender.model is not None
    }

@app.get('/recommend/content/{user_id}', response_model=RecommendationResponse)
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
            n_recommendations=10
        )
        
        # 컨텐츠 정보 조회 및 응답 생성
        result = []
        for rec in recommendations:
            content = db.query(Content).filter(Content.id == rec['content_id']).first()
            if content:
                result.append(ContentRecommendation(
                    id=content.id,
                    name=content.name,
                    thumbnail=content.thumbnail,
                    score=rec['score'],
                    cf_score=rec['cf_score'],
                    genre_score=rec['genre_score'],
                    distance_score=rec['distance_score']
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

@app.post('/recommend/retrain')
def retrain_model(db: Session = Depends(get_db)):
    """
    추천 모델 재학습 (관리자용 엔드포인트)
    """
    global recommender
    try:
        recommender = LightFMRecommender(db)
        return {"message": "Model retrained successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retraining failed: {str(e)}"
        )