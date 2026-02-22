from dataclasses import dataclass


@dataclass
class User:
    user_id: str
    email: str


def get_current_user() -> User:
    # Placeholder for HF OAuth integration
    return User(user_id="demo-user", email="demo@example.com")
