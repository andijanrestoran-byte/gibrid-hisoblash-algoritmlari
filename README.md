# GibridSim — gibrid hisoblash algoritmlari simulyatori

**GibridSim** — dissertatsiya mavzusi _"Gibrid hisoblash algoritmlarini
rivojlantirish va ularning differensial tenglamalar modellarida qo'llanilishi"_
doirasida ishlab chiqilgan Django veb-ilovasi. U **uzluksiz (ODE)** va
**diskret (hodisa/o'tish)** dinamikani birlashtirgan **gibrid tizimlarni**
sonli modellashtiradi va natijalarni interaktiv (Plotly) grafiklarda
vizualizatsiya qiladi.

---

## Asosiy imkoniyatlar (dissertatsiya natijalari)

1. **Gibrid avtomat yadrosi** (`simulator/core/hybrid_automaton.py`)
   Mode (uzluksiz holat, `f(t, x)`), Transition (guard `g(t, x)` + reset
   `R(t, x)`), gibrid vaqt hisobi va Zeno himoyasi (maksimal hodisalar soni).

2. **Hodisalarni aniqlash** (`simulator/core/event_detection.py`)
   Guard ishorasi o'zgarganda o'tish nuqtasi `scipy solve_ivp` hodisalari
   bilan, hamda taqqoslash uchun **mustaqil bisektsiya** usuli bilan
   `1e-9` aniqlikkacha lokalizatsiya qilinadi.

3. **AUTO_ODE — usulni avtomatik almashtirish** (`simulator/core/solver_manager.py`)
   Simulyatsiya RK45 (ochiq usul) bilan boshlanadi; masala **qat'iyligi**
   (Yakobian spektral radiusi) aniqlansa, avtomatik ravishda **Radau/BDF**
   (yashirin usul)ga o'tadi; soddalashsa qaytadi. Har bir intervalda qaysi
   usul ishlatilgani jurnalda ko'rsatiladi. Qo'lda tanlash ham mumkin:
   `AUTO / RK45 / Radau / BDF / LSODA`.

4. **Adaptiv grafik nuqtalari** (`simulator/core/plotting.py`)
   Yechim son jihatidan to'g'ri bo'lsa ham, kam tugun orasini to'g'ri chiziq
   bilan chizish keskin cho'qqilarni yo'qotishi mumkin. Adaptiv rejim zich
   chiqish (dense output) orqali hosila katta joylarda nuqta qo'shadi.
   **Naive** (≈ asinxron) va **adaptiv** (moslashuvchan to'r) qatorlar yonma-yon
   taqqoslanadi — dissertatsiyaning sinxron/asinxron grafik qurish g'oyasi (§2.3).

5. **DAE va indeksni qisqartirish** (`simulator/core/dae.py`, `structural.py`)
   Algebraik-differensial tenglamalar (`dx/dt=f(t,x,y)`, `0=g(t,x,y)`) yarim-oshkor
   indeks-1 ko'rinishda yechiladi: algebraik o'zgaruvchi har qadamda Nyuton usuli
   bilan topiladi. Yuqori indeksli masalalar (mayatnik — indeks-3) cheklovni
   differensiallash orqali indeks-1 ga keltiriladi. **Strukturaviy tahlil**
   (`structural.py`) insidentlik matritsasi va **Pantelides algoritmi** orqali
   indeksni avtomatik aniqlaydi (§2.1) — `pendulum_dae` sahifasidagi "Struktura" tabi.

> Yadro (`simulator/core/`) **Django'ga bog'liq emas** — sof Python kutubxona,
> uni mustaqil sinash va qayta ishlatish mumkin.

---

## Tayyor misollar

| Kalit | Nomi | Tavsifi |
|-------|------|---------|
| `bouncing_ball` | Sakrovchi to'p | `x'=v, v'=-g`; guard `x≤0 & v<0`; reset `v:=-e·v`. Zeno hodisasi. |
| `thermostat` | Termostat | Ikki rejim: `T'=-K·T` va `T'=K(h-T)`; guardlar `T≤m`, `T≥M`. Davriy. |
| `three_state` | Uch tugunli avtomat | Uch rejim siklik almashadi (`a=-1`, `b=[1,-1,1]`); qat'iy davriy yechim. |
| `sharp_peak` | Keskin cho'qqi | `dx/dt=f'(t)`, `f(t)=t+exp(-(t-1)²/0.01)`. Naive vs adaptiv grafik (T=2,4,12). |
| `smooth_linear` | Qat'iy bo'lmagan model | `x'=A·x`, λ=-1,-3 (silliq). AUTO ochiq RK45 da qoladi (stiff_demo ga qarama-qarshi). |
| `stiff_demo` | Qat'iy sistema | `x'=-A_n·x`, `A_n=[[1+2⁻ⁿ,1],[1,1+2⁻ⁿ]]`. n katta — AUTO yashirin usulga o'tadi. |
| `van_der_pol` | Van der Pol ostsilyatori | `y'=μ(1-x²)y-x` nochiziqli; μ katta — qat'iy relaksatsion tebranish, AUTO yashirin usulga o'tadi. |
| `cat_mouse` | Mushuk va sichqon | Ta'qib gibrid tizimi (1.3-rasm); hodisaga asoslangan o'tishlar (ushlash/qochish). |
| `rocket_pursuit` | Raketa ta'qibi | Diskret vaqtli (sample-data) ta'qib; yo'nalish faqat har Δt da yangilanadi. |
| `pendulum_dae` | Cheklovli mayatnik (DAE) | Indeks-3 → indeks-1 qisqartirish; `x²+y²=L²` cheklovi, taranglik λ algebraik. Strukturaviy tahlil tabi. |

Har bir misol o'zbekcha nomi, tavsifi, default parametrlari va forma
metama'lumotlariga ega (`simulator/presets/`).

---

## O'rnatish

Python **3.10+** kerak.

```bash
# 1. Loyiha papkasiga kiring
cd gibridsim

# 2. Virtual muhit yarating va faollashtiring
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

# 3. Bog'liqliklarni o'rnating
pip install -r requirements.txt

# 4. Bazani tayyorlang (SQLite)
python manage.py migrate

# 5. (ixtiyoriy) admin foydalanuvchi
python manage.py createsuperuser

# 6. Serverni ishga tushiring
python manage.py runserver
```

So'ng brauzerda **http://127.0.0.1:8000/** ni oching.

---

## Foydalanish

- **Bosh sahifa** — loyiha tavsifi va misollar kartochkalari.
- **/simulate/<misol>/** — chap tomonda parametrlar formasi (T, rtol, atol,
  usul, misol parametrlari) va **"Hisoblash"** tugmasi. O'ng tomonda tablar:
  1. **Vaqt grafigi** (Plotly) — o'tish nuqtalari vertikal chiziqlar bilan.
  2. **Fazaviy portret** (2 o'zgaruvchili modellarda).
  3. **Hodisalar jadvali** — vaqt, o'tish, reset oldi/keyin holat, bisektsiya.
  4. **Hisob jurnali** — qaysi intervalda qaysi usul ishlatilgani.
- **Naive/Adaptiv taqqoslash** tugmasi (`sharp_peak` uchun standart yoniq).
- **CSV yuklab olish** va Plotly orqali **PNG eksport** (grafik ustidagi kamera).
- Har bir hisob **SimulationRun** modelida saqlanadi va **/history/** sahifasida
  ko'rinadi (qayta ko'rish va CSV yuklab olish bilan).

Forma yuborilganda sahifa qayta yuklanmaydi: `fetch` orqali
`/api/simulate/<misol>/` JSON API chaqiriladi va grafiklar JS'da yangilanadi.

---

## Xavfsizlik

Foydalanuvchi kiritgan ifodalar **hech qachon `eval`/`exec` qilinmaydi**.
Parametrlar faqat son ko'rinishida formadan olinadi va chegaralar bo'yicha
validatsiya qilinadi. Tenglamalar faqat `simulator/presets/` modullarida kod
sifatida turadi.

---

## Testlar

```bash
pytest
```

Sinaladigan asosiy holatlar:

- Sakrovchi to'pda birinchi urilish vaqti analitik qiymat `√(2·x₀/g)` ga
  `1e-6` aniqlikda mos kelishi.
- `three_state` avtomatining davriyligi.
- `sharp_peak` da adaptiv rejim cho'qqini (≈2) yo'qotmasligi.
- `stiff_demo` da AUTO rejim usul almashtirishi (RK45 → Radau).
- `/api/simulate/` view'ining asosiy holatlari (200, validatsiya xatosi 400).

---

## Loyiha tuzilmasi

```
gibridsim/
  config/                  # Django sozlamalari, URL'lar, WSGI/ASGI
  simulator/
    core/                  # SOF PYTHON yadro (Django'siz)
      hybrid_automaton.py  # gibrid avtomat modeli
      event_detection.py   # o'tish nuqtalarini topish (bisektsiya)
      solver_manager.py    # AUTO_ODE — usulni avtomatik almashtirish
      dae.py               # algebraik-differensial tenglamalar (indeks-1)
      structural.py        # insidentlik matritsasi + Pantelides (indeks tahlili)
      plotting.py          # adaptiv grafik nuqtalari (JSON)
    presets/               # tayyor misollar (Python modullar)
    results.py             # natijani JSON/CSV ga seriyalash
    models.py              # SimulationRun (saqlangan natijalar)
    forms.py, views.py, urls.py
    templates/simulator/   # Bootstrap 5 + Plotly shablonlar
    static/simulator/      # app.js (fetch + Plotly)
    tests/                 # pytest testlari
  manage.py
  requirements.txt
  README.md
```

---

## Deploy (PythonAnywhere)

1. **Kodni yuklang** — PythonAnywhere'da Bash konsolida loyihani `git clone`
   qiling yoki ZIP yuklang.
2. **Virtual muhit**:
   ```bash
   mkvirtualenv --python=/usr/bin/python3.10 gibridsim
   pip install -r requirements.txt
   ```
3. **Web ilova**: _Web_ bo'limida **"Add a new web app" → Manual configuration
   (Django emas, Manual)** ni tanlang va virtualenv yo'lini ko'rsating.
4. **WSGI fayl**: PythonAnywhere bergan WSGI faylini tahrirlab, quyidagini
   yozing:
   ```python
   import os, sys
   path = "/home/<foydalanuvchi>/gibridsim"
   if path not in sys.path:
       sys.path.insert(0, path)
   os.environ["DJANGO_SETTINGS_MODULE"] = "config.settings"
   os.environ["DJANGO_DEBUG"] = "0"
   os.environ["DJANGO_ALLOWED_HOSTS"] = "<foydalanuvchi>.pythonanywhere.com"
   os.environ["DJANGO_SECRET_KEY"] = "<tasodifiy-uzun-maxfiy-kalit>"
   from config.wsgi import application
   ```
5. **Statik fayllar**:
   ```bash
   python manage.py collectstatic --noinput
   python manage.py migrate
   ```
   _Web → Static files_ da: URL `/static/` → papka
   `/home/<foydalanuvchi>/gibridsim/staticfiles`.
6. **Reload** tugmasini bosing.

> Boshqa hostinglar (Render, Railway, VPS + Gunicorn/Nginx) uchun ham xuddi
> shu tamoyil: `DJANGO_DEBUG=0`, `DJANGO_ALLOWED_HOSTS` va `DJANGO_SECRET_KEY`
> ni muhit o'zgaruvchilari orqali bering, `collectstatic` va `migrate` ni
> bajaring.

---

## Dissertatsiya bilan bog'liqlik

| Dissertatsiya g'oyasi | Ilovadagi amaliyot |
|----------------------|--------------------|
| Gibrid tizimlarni sonli modellashtirish (§1.2) | `core/hybrid_automaton.py` + `solver_manager.py` |
| O'tish nuqtalarini aniq lokalizatsiya (§2.2) | `event_detection.py` (bisektsiya 1e-9) |
| Qat'iylikka qarab usulni almashtirish — AUTO_ODE (§3.1) | `solver_manager.py` (RK45 ↔ Radau/BDF) |
| To'g'ri yechim ≠ to'g'ri grafik; sinxron/asinxron grafik (§2.3) | `plotting.py` (naive vs adaptiv), `sharp_peak` |
| Qat'iy sistemalar (§3.2) | `stiff_demo` (A_n) va `smooth_linear` (kontrast) |
| DAE va indeksni qisqartirish (§2.1) | `core/dae.py`, `core/structural.py` (Pantelides), `pendulum_dae` |
| Nochiziqli test modellari (§3.2) | `van_der_pol` |
| Ta'qib gibrid tizimlari (kirish, 1.3-rasm) | `cat_mouse`, `rocket_pursuit` (sample-data) |
| Uch tugunli davriy avtomat (1.1-jadval) | `three_state` |
| Termostat — harorat kontrolleri (1.2-rasm) | `thermostat` |

GibridSim bu nazariy natijalarni interaktiv, takrorlanadigan va vizual
ko'rinishda namoyish etadi.
