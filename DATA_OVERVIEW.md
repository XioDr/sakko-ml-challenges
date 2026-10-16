# دیکشنری و بررسی دقیق داده — هکتون Sakko/ICDS

این فایل خلاصه‌ی دقیق و تأییدشده (با اجرای واقعی pandas روی فایل‌ها، نه فقط PDF) از ساختار داده‌ی مشترک سه چالش است. اعداد missing%/nunique همه از خودِ داده در `data/` استخراج شده‌اند.

## 1. واحد نمونه و مفهوم پایه

- هر ردیف = یک **آیتم** از یک **اعلان کیفیت (Notification)**. یک اعلان می‌تواند چند آیتم/ردیف عیب داشته باشد.
- کلید یکتای هر نمونه **ترکیبی** است: `(Notification, Item)`. هیچ‌کدام به‌تنهایی یکتا نیست (`Item` فقط ۱۳۷ مقدار متمایز در کل train دارد چون شماره‌ی ردیف عیب داخل یک اعلان است، نه شناسه‌ی سراسری).
- هدف پیش‌بینی: مجموع زمان دوباره‌کاری ثبت‌شده برای آن آیتم بر حسب **دقیقه** (`rework_minutes`)؛ نه زمان تقویمی بسته شدن پرونده.
- داده از دو سامانه‌ی کارخانه‌ای جمع شده اما شناسه‌ی `Notification` بین دو سامانه تداخل ندارد (نیازی به ستون سیستم مبدأ برای join نیست).

## 2. تقسیم زمانی و اندازه‌ی فایل‌ها

| بخش | آیتم | اعلان | بازه‌ی Created on |
|---|---|---|---|
| آموزش | 73,239 | 19,701 | 2007-11-26 → 2024-12-30 |
| آزمون | 8,323 | 2,252 | 2024-12-31 → 2025-12-31 |

جدول فایل‌ها (`data/` و `data/train_test_features/`):

| فایل | ردیف train | ردیف test | واحد هر ردیف |
|---|---|---|---|
| `items_train/test.csv` | 73,239 | 8,323 | یک آیتم اعلان |
| `target_binary_train.csv` | 73,239 | — | فقط آموزش |
| `target_multiclass_train.csv` | 73,239 | — | فقط آموزش |
| `target_regression_train.csv` | 73,239 | — | فقط آموزش |
| `first_activity_train/test.csv` | 31,087 | 4,146 | نخستین فعالیت واجد شرایط هر آیتم |
| `item_longtext_*_embeddings.npy`+keys | 72,908 | 7,621 | بردار ۴۰۹۶بعدی شرح آیتم |
| `first_activity_*_embeddings.npy`+keys | 24,479 | 2,828 | بردار ۴۰۹۶بعدی شرح فعالیت |
| `mas_trees_train/test.csv` | 52,931 (7,990 درخت) | 7,688 (940 درخت) | یک گره‌ی سفارش تولید |
| `material_general_train/test.csv` | 3,302 | 2,226 | یک رکورد پایه‌ی قلم |
| `material_desc_*_embeddings.npy`+keys | 3,302 | 2,226 | بردار ۴۰۹۶بعدی شرح قلم |

> نکته‌ی مهم: بردارهای متنی **۴۰۹۶بعدی**اند (نه ۶۹۰۴ که در PDF به‌خاطر خطای OCR اشتباه استخراج شده بود — با `np.load` تأیید شد).

## 3. روابط و کلیدهای اتصال

```
items (Notification, Item)  ──1:1──  target_*  (فقط train)
items                       ──1:0..1──  first_activity   [کلید: Notification, Item]
items                       ──1:0..1──  item_longtext keys  [کلید: notif=Notification, item=Item]
first_activity              ──1:0..1──  first_activity keys [کلید: notif, item, activity, delay_h]
items.tree_id                ──many:many──  mas_trees.tree_id   (قبل از join باید تجمیع شود)
items.Material               ──many:0..1──  material_general.Material
items.Assembly                ──many:0..1──  material_general.Material   (join دوم و جداگانه)
mas_trees.Material            ──many:0..1──  material_general.Material
mas_trees.Parent Material     ──many:0..1──  material_general.Material
material_general.Material     ──1:1──  material_desc keys
```

قواعد کلیدی از دیکشنری رسمی:
- items را جدول پایه بگیرید و بقیه را left-join کنید؛ اتصال مستقیم items به همه‌ی گره‌های یک درخت، ردیف تکراری/نمونه‌ی جدید می‌سازد — **قبل از اتصال، mas_trees را بر اساس `tree_id` تجمیع کنید**.
- وجود `Material`/`Assembly` تضمین‌کننده‌ی وجود رکورد متناظر در `material_general` نیست (زیر را ببینید).
- یک اعلان/درخت می‌تواند هم در train هم در test دیده شود؛ این به‌تنهایی نشانه‌ی خطا نیست.

## 4. جدول `items` — ستون به ستون (۲۴ ستون)

| ستون | dtype | missing % train/test | nunique train/test | توضیح |
|---|---|---|---|---|
| `Notification` | id | 0 / 0 | 19,701 / 2,252 | شناسه‌ی اعلان کیفیت |
| `Item` | id | 0 / 0 | 137 / 128 | شماره‌ی ردیف عیب داخل اعلان؛ به‌تنهایی یکتا نیست |
| `Created on` | int YYYYMMDD | 0 / 0 | 5,644 / 346 | تاریخ ایجاد آیتم؛ مبنای تفکیک train/test |
| `Created At` | int HHMMSS (صفر ابتدایی حذف‌شده) | 0 / 0 | 35,148 / 6,398 | ساعت ایجاد؛ قبل تبدیل به زمان صفر ابتدایی را برگردانید |
| `Problem group text` | cat (18 مقدار) | 0.36 / 0.16 | — | گروه کلی عیب |
| `Problem Code Text` | cat (110) | 0.36 / 0.16 | — | کد دقیق عیب؛ ممکن است با بیش از یک گروه دیده شود |
| `Obj. part grp. txt.` | cat (69) | 0.49 / 0.35 | — | گروه کلی محل عیب |
| `Obj. part code text` | cat (224) | 0.49 / 0.35 | — | کد دقیق محل عیب |
| `Cause group text` | cat (20) | 0.20 / 1.01 | — | گروه کلی علت |
| `Cause code text` | cat (83) | 0.20 / 1.01 | — | کد دقیق علت؛ رابطه‌ی علّی قطعی نیست |
| `Severity` | cat (11: severity_02..13) | ~0.01 / 0.0 | — | **ویژگی ورودی است، هدف نیست**؛ بدون ترتیب فرض‌شده |
| `Priority Text` | cat (4: priority_1..4) | **55.4 / 43.3** | — | برای بخش زیادی از آیتم‌ها ثبت نشده |
| `Number of Defects` | int | 0 / 0 | 134 / 54 | median=1، max=2,044,372 (دم بسیار سنگین) |
| `DefectiveQty (Internal)` | float | 0 / 0 | 17 / 28 | بیشتر مقادیر صفر؛ max=960 |
| `DefectiveQty (External)` | float | 0 / 0 | 5 / 13 | بیشتر مقادیر صفر؛ max=960 |
| `Complaint Quantity` | float | **90.7 / 3.6** | 103 / 122 | پوشش train/test خیلی متفاوت است |
| `Reference Quantity` | float | **90.7 / 3.6** | 115 / 123 | الگوی missing دقیقاً مثل Complaint Quantity |
| `Assembly` | id | 1.0 / 0.55 | 6,451 / 1,683 | شناسه‌ی مجموعه‌ی مرتبط؛ فقط ۱۴۰۵ از ۶۴۵۱ مقدار متمایز train در material_general هست |
| `Material` | id | 1.12 / 0.0 | 2,104 / 850 | شناسه‌ی قلم/قطعه؛ ۲۰۲۳ از ۲۱۰۴ مقدار متمایز train در material_general هست (اما با وزن فراوانی ردیف، فقط ~74% ردیف‌ها match می‌شوند) |
| `Unit of Measure` | cat | 0.77 / 0.56 | 2 / 1 | تقریباً همیشه `uom_2` |
| `Profit Center` | cat (12) | 90.7 / 3.6 | — | الگوی missing مثل Complaint/Reference/WBS |
| `Prevention Factor` | cat (10) | 0 / 0 | — | کد پیشگیری؛ مقیاس/ترتیب تعریف‌نشده |
| `WBS element` | — | 90.7 / 3.6 | 1 مقدار غیر-خالی | عملاً بی‌فایده؛ در پایپ‌لاین حذف شده |
| `tree_id` | id | 0 / 0 | 7,990 / 940 | لینک به mas_trees؛ گره‌ی یکتا را مشخص نمی‌کند |

نکته‌ی داده‌ای مهم: چهار ستون `Complaint Quantity`, `Reference Quantity`, `Profit Center`, `WBS element` دقیقاً همان الگوی missing (۹۰.۷%/۳.۶%) را دارند → به‌احتمال زیاد از یک زیرسیستم/فرم مشترک می‌آیند که فقط برای زیرمجموعه‌ای از آیتم‌ها پر می‌شود.

## 5. جدول `first_activity` (پوشش ۴۲.۵% train / ۴۹.۸% test)

| ستون | نکته |
|---|---|
| `Notification`, `Item` | کلید اتصال به items |
| `activity` | شناسه‌ی ترتیبی فعالیت در اعلان (175 مقدار متمایز)؛ مدت یا تعداد فعالیت نیست |
| `delay_h` | فاصله‌ی ایجاد آیتم تا فعالیت، ساعت؛ بازه‌ی [0, 24)، median≈4.9، mean≈9.1 |
| `Code Text for Activity` | ۱۴ مقدار (act_code_09 با 14,434 مورد پرتکرارترین)؛ 0.2% خالی |
| `Code Group Text for Activity` | ۴ مقدار؛ 99% مربوط به `act_code_group_1` |

نبود ردیف فقط یعنی «فعالیت واجد شرایط استخراج نشده»، نه لزوماً «بدون رسیدگی».

## 6. جدول `mas_trees` (درخت ساخت‌وساز/مونتاژ)

- 52,931 گره در 7,990 درخت (train) و 7,688 گره در 940 درخت (test).
- `Depth`: 0 تا 10 (ریشه=0).
- `is_root`, `in_mas`: پرچم دودویی (گره‌ی جایگزین وقتی سفارش در استخراج MAS نبوده `in_mas=0` و تک‌گره‌ای با `is_root=1, Depth=0` است).
- `AssignmentType`: عملاً دو مقدار (`assignment_1` غالب، `assignment_2` نادر)، برای ریشه‌ها خالی (15.1%).
- `Parent Order` 15.1%، `Material` 17.1%، `Parent Material` 33.7% خالی.
- تعداد گره در هر tree: median=1 ولی mean≈6.6، max=346 — قبل از join حتماً تجمیع شود.

## 7. جدول `material_general` (فقط 3,302 ماده در train، 2,226 در test)

| ستون | نکته |
|---|---|
| `Material_type` | تقریباً همیشه `mat_type_1` (3259/3302) |
| `Base_Unit` | تقریباً همیشه `base_unit_2` |
| `QM_Procurement` | 57.2% خالی؛ خالی را «منفی» فرض نکنید |
| `Material_Group` | 0.3% خالی، 2 مقدار |
| `Gross_weight`, `Net_weight`, `Volume` | عددی؛ صفرهای ثبت‌شده واقعی‌اند، به missing تبدیل نکنید |
| `Weight_unit` | تقریباً همیشه `weight_unit_2` |
| `Volume_unit` | 96.3% خالی |
| `Size_dimensions` | 99.3% خالی، بدون الگوی قابل parse |
| `Base_Material` | 58.0% خالی |
| `Expiration_Date` | 3.5% خالی؛ **کد وضعیت است نه تاریخ تقویمی** — تبدیل به date نکنید |

## 8. بردارهای متنی (embeddings)

سه خانواده، همه ۴۰۹۶بعدی float32، هرکدام یک فایل کلید parquet مرتبط:

| خانواده | کلید | پوشش train/test |
|---|---|---|
| `item_longtext` | `SAP System, notif, item` | 72,908 / 7,621 (~99.5% از items) |
| `first_activity` | `SAP System, notif, item, activity, delay_h` | 24,479 / 2,828 (زیرمجموعه‌ی first_activity، چون شرح بلند برای همه‌ی فعالیت‌ها نبوده) |
| `material_desc` | `Material` | 3,302 / 2,226 (دقیقاً هم‌اندازه‌ی material_general) |

ترتیب ردیف matrix دقیقاً با ترتیب ردیف keys یکی است؛ مرتب‌سازی جداگانه‌ی هرکدام رابطه را از بین می‌برد. متن‌های یکسان می‌توانند بردار یکسان تولید کنند (تکراری بودن بردار دلیل حذف نیست).

## 9. فایل‌های هدف (فقط train)

هر سه از یک «مجموع زمان ثبت‌شده برای دوباره‌کاری» ساخته شده‌اند؛ صفر برای آیتم بسته‌شده معتبر است.

| چالش | فایل | ستون هدف | ستون خروجی موردنیاز | توزیع |
|---|---|---|---|---|
| ۱ (باینری) | `target_binary_train.csv` | `rework_length` | `pred_rework_length` | short=50,086 (68.4%) / long=23,153 (31.6%) — آستانه ۴۸۰ دقیقه |
| ۲ (۴کلاسه) | `target_multiclass_train.csv` | `rework_bin` | `pred_rework_bin` | quick_le_2h=26,079 / same_day_2_8h=24,007 / multi_shift_8_24h=13,038 / multi_day_gt_24h=10,115 |
| ۳ (رگرسیون) | `target_regression_train.csv` | `Actual Work Minute` | `pred_rework_minutes` | ستون خام (بدون کلیپ)؛ min=0، median=240، max=7,800,180؛ فقط ۲,۰۰۵ ردیف دقیقاً صفر |

نکته‌ی حیاتی چالش ۳: مقدار خام رگرسیون کلیپ‌نشده نگه داشته می‌شود، اما **در ارزیابی**، هم مقدار واقعی و هم پیش‌بینی به بازه‌ی [۱۲۰، ۲۰۰۰] دقیقه محدود (clip) می‌شوند — مدل را روی مقیاس خام آموزش دهید ولی اثر این clip روی معیار را در نظر بگیرید.

برای join برچسب به آیتم‌ها فقط از `(Notification, Item)` استفاده کنید؛ این شناسه‌ها لازم نیست در فایل پیش‌بینی نهایی وجود داشته باشند — ترتیب و تعداد ردیف پیش‌بینی باید دقیقاً با test یکی باشد.

## 10. تله‌های داده‌ای که هنگام EDA/ساخت فیچر تأیید شد

- `material_found`≈74% و اتصال از طریق `Assembly`≈22% — چون material_general فقط 3,302 قلم دارد و دقیقاً همان Material های پرتکرار در items بیرون این لیست‌اند (شمارش «مقدار متمایز» گمراه‌کننده است؛ باید وزن‌دار به تعداد ردیف نگاه کرد).
- `has_first_activity`≈42.5% train / 49.8% test — با دیکشنری هم‌خوان.
- چهار ستون Complaint/Reference/Profit Center/WBS دقیقاً هم‌الگو missing دارند (۹۰.۷%/۳.۶%) → یک پرچم missing مشترک برایشان کافی است، نه چهار پرچم جدا.
- `Number of Defects` و مقادیر رگرسیون دم بسیار سنگین دارند (max میلیونی در برابر median تک‌رقمی/سه‌رقمی) → از `log1p` استفاده شود، نه مقیاس خطی.
- صفر عددی (وزن/حجم/DefectiveQty) را با missing یکی نگیرید؛ خالی یعنی «ثبت نشده»، صفر یعنی «ثبت شده به‌عنوان صفر».

## 11. فیچرهای ساخته‌شده تا این لحظه

پایپ‌لاین `shared_features/build_features.py` این جدول‌ها را بدون استفاده از هیچ target ترکیب کرده و ۹۶ فیچر تولید می‌کند (لیست کامل در [shared_features/feature_list.txt](shared_features/feature_list.txt) و شرح دسته‌بندی‌ها در [shared_features/README.md](shared_features/README.md)). سابقه‌ی target آیتم‌های خواهر زیر یک Notification عمداً اینجا نیامده چون نیاز به رفتار leakage-safe مخصوص هر چالش دارد.
