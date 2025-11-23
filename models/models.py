# models.py

from typing import List, Optional
from pydantic import BaseModel

# User 엔티티를 FastAPI에서 사용할 Pydantic 모델로 정의
class User(BaseModel):
    id: int
    nickname: Optional[str] = None
    address: Optional[str] = None

    class Config:
        # ORM 모델을 Pydantic 모델로 변환할 수 있도록 허용
        orm_mode = True


class ContentRecommendation(BaseModel):
    """추천 결과의 컨텐츠 정보"""
    id: int
    name: str
    isLiked: Optional[bool] = None
    thumbnail: Optional[str] = None


class RecommendationRequest(BaseModel):
    user_id: int

class RecommendationResponse(BaseModel):
    placeList: List[ContentRecommendation]

class PlaceRecommendation(BaseModel):
    id: int
    name: str
    isLiked: Optional[bool] 
    thumbnail: Optional[str] = None
    rating: Optional[float] 
    reviewCount: Optional[int] 
    latitude: Optional[float] 
    longitude: Optional[float] 
    content: Optional[List[str]] = None
    
class MapRecommendationResponse(BaseModel):
    placeList: List[PlaceRecommendation]


