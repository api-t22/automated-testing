from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TestData(BaseModel):
    """Test data for form inputs and verification."""
    expected_url: Optional[str] = None
    expected_text: Optional[str] = None
    expected_failure: Optional[bool] = False  # True for negative tests

    # Dynamic fields for form inputs (username, email, password, etc.)

    class Config:
        extra = "allow"  # Allows additional fields like username, password, etc.


class TestCase(BaseModel):
    id: str = Field(..., example="TC-001")
    feature: str
    title: str
    steps: List[str]
    expected_result: str
    priority: str
    type: str
    tags: List[str] = []
    risk: Optional[str] = None
    trace: Optional[str] = None
    preconditions: List[str] = []
    negative_cases: List[str] = []
    test_data: Optional[TestData] = None
    platform_matrix: Optional[Dict[str, Any]] = None


class UserStory(BaseModel):
    id: str = Field(..., example="US-001")
    feature: str
    title: str
    role: str
    goal: str
    benefit: str
    acceptance_criteria: List[str] = []
    related_test_ids: List[str] = []


class TestPlanResponse(BaseModel):
    document_title: Optional[str] = None
    summary: Optional[str] = None
    user_stories: List[UserStory] = []
    test_cases: List[TestCase]
    assumptions: List[str] = []
    risks: List[str] = []
