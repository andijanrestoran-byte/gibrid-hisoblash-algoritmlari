"""WSGI kirish nuqtasi (deploy uchun).

Vercel kabi serverless muhitda fayl tizimi faqat `/tmp` ga yoziladi. Shu sababli
sovuq startda bazani tayyorlaymiz (migrate) va statik fayllarni yig'amiz
(collectstatic) — ikkalasi ham `/tmp` ga (sozlama: settings.py VERCEL bloki).
Statik yig'ish WhiteNoise middleware STATIC_ROOT ni o'qishidan OLDIN bajarilishi
shart, shuning uchun `get_wsgi_application()` dan oldin chaqiriladi.
"""

import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

from django.core.wsgi import get_wsgi_application

if os.environ.get("VERCEL"):
    import django

    django.setup()
    from django.core.management import call_command

    try:
        call_command("migrate", "--noinput")
    except Exception as exc:  # baza tayyor bo'lmasa ham ilova ishlasin
        print(f"[vercel] migrate o'tkazib yuborildi: {exc}")
    try:
        call_command("collectstatic", "--noinput")
    except Exception as exc:
        print(f"[vercel] collectstatic o'tkazib yuborildi: {exc}")

application = get_wsgi_application()
# Vercel @vercel/python `app` yoki `application` nomli WSGI ilovasini qidiradi.
app = application
