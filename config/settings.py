"""GibridSim Django sozlamalari.

Sodda, bitta mashinada ishlaydigan konfiguratsiya: SQLite bazasi, sinxron
simulyatsiya (Celery/Redis yo'q), Bootstrap + Plotly frontend.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Xavfsizlik ---------------------------------------------------------------
# DIQQAT: bu kalit faqat ishlab chiqish uchun. Deploy paytida muhit
# o'zgaruvchisi orqali yangi maxfiy kalit bering.
import os

SECRET_KEY = os.environ.get(
    "DJANGO_SECRET_KEY", "dev-gibridsim-faqat-ishlab-chiqish-uchun-almashtiring"
)
DEBUG = os.environ.get("DJANGO_DEBUG", "1") == "1"
ALLOWED_HOSTS = os.environ.get("DJANGO_ALLOWED_HOSTS", "*").split(",")

# Render (yoki boshqa hosting) avtomatik bergan domenni qo'shamiz va POST
# so'rovlari (CSRF) shu domendan kelishiga ruxsat beramiz.
CSRF_TRUSTED_ORIGINS = []
RENDER_HOST = os.environ.get("RENDER_EXTERNAL_HOSTNAME")
if RENDER_HOST:
    ALLOWED_HOSTS.append(RENDER_HOST)
    CSRF_TRUSTED_ORIGINS.append(f"https://{RENDER_HOST}")
# *.onrender.com domenidan kelgan so'rovlarga ham ishonamiz (zaxira).
CSRF_TRUSTED_ORIGINS.append("https://*.onrender.com")

# --- Ilovalar -----------------------------------------------------------------
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "simulator",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    # WhiteNoise — statik fayllarni Django o'zi (gunicorn ortida) tarqatadi.
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"

# --- Baza -------------------------------------------------------------------
# DATABASE_URL berilsa (masalan, Render PostgreSQL) — o'sha ishlatiladi,
# aks holda lokal SQLite. Deploy'da SQLite vaqtinchalik bo'lishi mumkin
# (xizmat qayta ishga tushganda "Tarix" tozalanadi) — bu demo uchun maqbul;
# doimiy saqlash kerak bo'lsa, Render'da bepul PostgreSQL ulang.
import dj_database_url

DATABASES = {
    "default": dj_database_url.config(
        default=f"sqlite:///{BASE_DIR / 'db.sqlite3'}",
        conn_max_age=600,
    )
}

# --- Parol validatsiyasi ------------------------------------------------------
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
]

# --- Til va vaqt --------------------------------------------------------------
LANGUAGE_CODE = "uz"
TIME_ZONE = "Asia/Tashkent"
USE_I18N = True
USE_TZ = True

# --- Statik fayllar -----------------------------------------------------------
STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

# WhiteNoise: statik fayllarni siqib (gzip/brotli) tarqatadi. Manifest emas —
# biror statik havola yetishmasa ham 500 bermaydi (demo uchun xavfsizroq).
STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    "staticfiles": {
        "BACKEND": "whitenoise.storage.CompressedStaticFilesStorage",
    },
}

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Simulyatsiya xavfsizlik chegaralari (cheksiz hisobni oldini olish uchun).
GIBRIDSIM_MAX_T = 1000.0
