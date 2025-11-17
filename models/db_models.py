# models/db_models.py

from sqlalchemy import Column,BigInteger, Integer, String, DateTime, ForeignKey, Float, Date, Text
from sqlalchemy.orm import relationship

from utils.database import Base


class User(Base):
    __tablename__ = "users"
    
    id = Column(BigInteger, primary_key=True, index=True)
    inactive_date = Column(Date, nullable=True)
    oauth_id = Column(BigInteger, nullable=True)
    nickname = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False)
    address = Column(String(255), nullable=True)
    
    # 관계
    content_likes = relationship("UserContentLikes", back_populates="user")
    place_likes = relationship("UserPlaceLikes", back_populates="user")
    categories = relationship("UserCategory", back_populates="user")


class Category(Base):
    """
    카테고리 마스터 테이블 (장르)
    예: 액션, 로맨스, SF, 코미디 등
    """
    __tablename__ = "category"
    
    id = Column(BigInteger, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    
    # 관계
    user_categories = relationship("UserCategory", back_populates="category")
    content_categories = relationship("ContentCategory", back_populates="category")


class UserCategory(Base):
    """
    사용자-카테고리 연결 테이블 (사용자 선호 장르)
    """
    __tablename__ = "user_category"
    
    id = Column(BigInteger, primary_key=True, index=True)
    user_id = Column(BigInteger, ForeignKey("users.id"), index=True, nullable=False)
    category_id = Column(BigInteger, ForeignKey("category.id"), index=True, nullable=False)
    
    # 관계
    user = relationship("User", back_populates="categories")
    category = relationship("Category", back_populates="user_categories")


class Content(Base):
    __tablename__ = "content"
    
    id = Column(BigInteger, primary_key=True, index=True)
    title_kr = Column(String(255), nullable=False)
    title_eng = Column(String(255), nullable=True)
    year = Column(Integer, nullable=True)
    plot = Column(Text, nullable=True)
    thumbnail = Column(Text, nullable=True)
    type = Column(Integer, nullable=False)  # 0: 영화, 1: TV 등
    runtime = Column(Integer, nullable=True)
    tmdb_id = Column(Integer, nullable=True, unique=True)
    
    # 관계
    likes = relationship("UserContentLikes", back_populates="content")
    categories = relationship("ContentCategory", back_populates="content")
    places = relationship("ContentPlace", back_populates="content")


class ContentCategory(Base):
    """
    컨텐츠-카테고리 연결 테이블 (컨텐츠의 실제 장르)
    """
    __tablename__ = "content_category"
    
    id = Column(BigInteger, primary_key=True, index=True)
    content_id = Column(BigInteger, ForeignKey("content.id"), index=True, nullable=False)
    category_id = Column(BigInteger, ForeignKey("category.id"), index=True, nullable=False)
    
    # 관계
    content = relationship("Content", back_populates="categories")
    category = relationship("Category", back_populates="content_categories")


class Place(Base):
    """
    촬영 장소 테이블
    """
    __tablename__ = "places"
    
    id = Column(BigInteger, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    
    # 관계
    likes = relationship("UserPlaceLikes", back_populates="place")
    contents = relationship("ContentPlace", back_populates="place")


class ContentPlace(Base):
    """
    컨텐츠-장소 연결 테이블 (촬영지 정보)
    """
    __tablename__ = "stillcut"
    
    id = Column(BigInteger, primary_key=True, index=True)
    content_id = Column(BigInteger, ForeignKey("content.id"), index=True, nullable=False)
    place_id = Column(BigInteger, ForeignKey("places.id"), index=True, nullable=False)
    
    # 관계
    content = relationship("Content", back_populates="places")
    place = relationship("Place", back_populates="contents")


class UserContentLikes(Base):
    """
    유저가 컨텐츠에 대해 찜을 기록하는 테이블
    """
    __tablename__ = "content_likes"
    
    id = Column(BigInteger, primary_key=True, index=True)
    user_id = Column(BigInteger, ForeignKey("users.id"), index=True, nullable=False)
    content_id = Column(BigInteger, ForeignKey("content.id"), index=True, nullable=False)
    
    # 관계
    user = relationship("User", back_populates="content_likes")
    content = relationship("Content", back_populates="likes")


class UserPlaceLikes(Base):
    """
    유저가 장소에 대해 찜을 기록하는 테이블
    """
    __tablename__ = "place_likes"
    
    id = Column(BigInteger, primary_key=True, index=True)
    user_id = Column(BigInteger, ForeignKey("users.id"), index=True, nullable=False)
    place_id = Column(BigInteger, ForeignKey("places.id"), index=True, nullable=False)
    
    # 관계
    user = relationship("User", back_populates="place_likes")
    place = relationship("Place", back_populates="likes")