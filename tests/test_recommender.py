# tests/test_recommender_mock.py

import pytest
from unittest.mock import MagicMock

from recommender import LightFMRecommender
from models.db_models import User, Content, UserContentLikes, Category, ContentCategory, UserCategory


class FakeQuery:
    """SQLAlchemy의 .query().filter().all() 체인을 흉내내기 위한 Mock Query 객체"""
    def __init__(self, data):
        self.data = data
        self._filter_fn = None

    def all(self):
        if self._filter_fn:
            return list(filter(self._filter_fn, self.data))
        return self.data

    def filter(self, fn):
        self._filter_fn = lambda obj: fn(obj)
        return self


class FakeDBSession:
    """SQLAlchemy Session을 흉내내는 Fake Session"""
    def __init__(self, datasets):
        self.datasets = datasets

    def query(self, model):
        return FakeQuery(self.datasets.get(model, []))


@pytest.fixture
def mock_db():
    """
    LightFMRecommender를 위한 Fake DB 준비
    """
    # Users
    users = [
        User(id=1, address="Seoul"),
        User(id=2, address="Busan"),
    ]

    # Contents
    contents = [
        Content(id=10, title_kr="Movie A", title_eng="Movie A", year=2010, type="movie"),
        Content(id=20, title_kr="Movie B", title_eng="Movie B", year=2020, type="movie"),
    ]

    # Likes
    interactions = [
        UserContentLikes(user_id=1, content_id=10),
        UserContentLikes(user_id=2, content_id=20),
    ]

    # Categories
    categories = [
        Category(id=1, name="Action"),
        Category(id=2, name="Drama"),
    ]

    # ContentCategory
    content_categories = [
        ContentCategory(content_id=10, category_id=1),
        ContentCategory(content_id=20, category_id=2),
    ]

    # UserCategory
    user_categories = [
        UserCategory(user_id=1, category_id=1),
    ]

    datasets = {
        User: users,
        Content: contents,
        UserContentLikes: interactions,
        Category: categories,
        ContentCategory: content_categories,
        UserCategory: user_categories,
    }

    return FakeDBSession(datasets)


def test_lightfm_recommender_train_and_predict(mock_db):
    """
    LightFMRecommender 학습 & 예측이 정상적으로 작동하는지 테스트
    """
    recommender = LightFMRecommender(db=mock_db)

   
    assert recommender.model is not None

 
    recs = recommender.predict_for_user(1, n_items=5)

    assert isinstance(recs, list)
    assert len(recs) >= 0 
    
    if recs:
        content_id, score = recs[0]
        assert isinstance(content_id, int)
        assert isinstance(score, float)
