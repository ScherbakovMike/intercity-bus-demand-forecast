"""Authentication endpoints. In-memory user store for simplicity + тестирования."""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from api.auth import create_access_token, get_current_user, hash_password, verify_password
from api.schemas import Token, UserCreate

router = APIRouter()

# Simple in-memory users (prod: Postgres via db.models.AppUser)
_USERS: dict[str, dict] = {
    "admin":     {"password_hash": hash_password("admin123"),     "role": "admin",      "full_name": "Администратор системы"},
    "planner":   {"password_hash": hash_password("planner123"),   "role": "planner",    "full_name": "Иван Планировщик"},
    "analyst":   {"password_hash": hash_password("analyst123"),   "role": "analyst",    "full_name": "Анна Аналитик"},
    "dispatcher":{"password_hash": hash_password("dispatch123"),  "role": "dispatcher", "full_name": "Олег Диспетчер"},
}


@router.post("/register")
def register(user: UserCreate, current: dict = Depends(get_current_user)):
    if current.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Только admin может регистрировать пользователей")
    if user.username in _USERS:
        raise HTTPException(status_code=400, detail="Пользователь уже существует")
    _USERS[user.username] = {
        "password_hash": hash_password(user.password),
        "role": user.role,
        "full_name": user.full_name or user.username,
    }
    return {"status": "created", "username": user.username, "role": user.role}


@router.post("/login", response_model=Token)
def login(form: OAuth2PasswordRequestForm = Depends()):
    user = _USERS.get(form.username)
    if not user or not verify_password(form.password, user["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Неверные учётные данные")
    token = create_access_token(sub=form.username, role=user["role"])
    return Token(access_token=token, role=user["role"])


@router.get("/me")
def me(current: dict = Depends(get_current_user)):
    user = _USERS.get(current["sub"])
    return {
        "username": current["sub"],
        "role": current.get("role"),
        "full_name": user.get("full_name") if user else None,
    }
