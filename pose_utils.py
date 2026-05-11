# pose_utils.py
import math
import numpy as np

# -----------------------
# 1. POSES DATABASE (Full Descriptions: EN / RU / KG)
# -----------------------
POSES_DB = {
    "tadasana": {
        "EN": {
            "title": "Tadasana (Mountain Pose)",
            "technique": (
                "1. Stand with feet together or hip-width apart, weight balanced evenly between both feet.\n"
                "2. Spread toes and press the four corners of each foot into the mat.\n"
                "3. Engage thighs, lift kneecaps slightly and draw the tailbone down.\n"
                "4. Lengthen through the spine, lift the chest but keep the ribs soft.\n"
                "5. Relax the shoulders away from the ears, arms hanging beside the torso or palms facing the thighs.\n"
                "6. Gaze forward or slightly upward. Breathe steadily for 5–10 breaths."
            ),
            "benefits": (
                "• Improves posture and body awareness.\n"
                "• Strengthens thighs, knees and ankles.\n"
                "• Helps develop balance and grounding."
            ),
            "contra": "Use support or hold onto a chair if you have severe balance problems.",
            "image": "assets/tadasana.png"
        },
        "RU": {
            "title": "Тадасана (Поза Горы)",
            "technique": (
                "1. Встаньте, поставив стопы вместе или на ширине тазобедренных суставов, равномерно распределив вес между обеими ногами.\n"
                "2. Расправьте пальцы ног и прижмите четыре точки каждой стопы к коврику.\n"
                "3. Активируйте мышцы бедер, слегка подтяните коленные чашечки и направьте копчик вниз.\n"
                "4. Вытягивайтесь через позвоночник, приподнимая грудную клетку, сохраняя при этом мягкость в рёбрах.\n"
                "5. Расслабьте плечи, отведите их от ушей; руки свободно опущены вдоль корпуса или ладони обращены к бёдрам.\n"
                "6. Смотрите вперёд или немного вверх. Дышите ровно в течение 5–10 дыханий."
            ),
            "benefits": (
                "• Улучшает осанку и осознанность тела.\n"
                "• Укрепляет бедра, колени и голеностопы.\n"
                "• Способствует развитию баланса и устойчивости."
            ),
            "contra": "При выраженных нарушениях равновесия используйте опору или держитесь за стул.",
            "image": "assets/tadasana.png"
        },
        "KG": {
            "title": "Тадасана (Тоодон турган поза)",
            "technique": (
                "1. Буттарды бириктирип же жамбаштын туурасында коюп түз туруңуз, салмакты эки бутка тең бөлүңүз.\n"
                "2. Манжаларды жайып, буттун төрт бурчун тең килемчеге бекем басыңыз.\n"
                "3. Сан булчуңдарын иштетип, тизе капкактарын бир аз көтөрүп, куймулчакты ылдый тартыңыз.\n"
                "4. Омуртканы узартып, көкүрөктү көтөрүңүз, бирок кабыргаларды бош кармаңыз.\n"
                "5. Ийиндерди кулактардан алыстатып бошотуңуз, колдор тулку бойдун жанында эркин салбырап турсун же алакандар сандардын тарабын карасын.\n"
                "6. Алдыга же бир аз жогору караңыз. 5–10 дем алуу тынч жана туруктуу алыңыз."
            ),
            "benefits": (
                "• Дененин түз абалын (осанкасын) жана дене аң-сезимин жакшыртат.\n"
                "• Сан булчуңдарын, тизелерди жана билектерди чыңдайт.\n"
                "• Тең салмакты жана жерге туташуу сезимин өнүктүрөт."
            ),
            "contra": "Эгер тең салмакта олуттуу кыйынчылыктар болсо, тирөөч колдонуңуз же отургучтан кармалып туруңуз.",
            "image": "assets/tadasana.png"
        }
    },

    "vriksasana": {
        "EN": {
            "title": "Vriksasana (Tree Pose)",
            "technique": (
                "1. Start in Tadasana. Shift weight onto the left foot.\n"
                "2. Bend the right knee and place the sole of the right foot on the inner left thigh or calf (avoid the knee joint).\n"
                "3. Press the standing foot firmly into the mat and the lifted foot into the thigh to create mutual resistance.\n"
                "4. Find a steady gaze (drishti) and balance. Option to bring hands to prayer at the chest or extend arms overhead.\n"
                "5. Keep hips level and spine long. Hold 5–10 breaths, then switch sides."
            ),
            "benefits": (
                "• Improves balance and concentration.\n"
                "• Strengthens ankles and legs; opens hips.\n"
                "• Builds mental focus and stability."
            ),
            "contra": "Avoid placing the foot on the knee joint. Use a wall for support if balance is unstable.",
            "image": "assets/vriksasana.png"
        },
        "RU": {
            "title": "Врикшасана (Поза Дерева)",
            "technique": (
                "1. Начните из позы Тадасана. Перенесите вес тела на левую ногу.\n"
                "2. Согните правое колено и поставьте подошву правой стопы на внутреннюю поверхность левого бедра или голени (избегайте коленного сустава).\n"
                "3. Плотно прижимайте опорную стопу к коврику и прижимайте поднятую стопу к бедру, создавая взаимное сопротивление.\n"
                "4. Найдите устойчивую точку для взгляда и сохраняйте равновесие. Руки можно сложить в намасте у груди или вытянуть вверх.\n"
                "5. Держите таз на одном уровне и вытягивайте позвоночник. Удерживайте позу 5–10 дыханий, затем смените сторону."
            ),
            "benefits": (
                "• Улучшает баланс и концентрацию.\n"
                "• Укрепляет голеностопы и ноги, раскрывает тазобедренные суставы.\n"
                "• Развивает ментальную устойчивость и сосредоточенность."
            ),
            "contra": "Не ставьте стопу на коленный сустав. При неустойчивом балансе используйте стену как опору.",
            "image": "assets/vriksasana.png"
        },
        "KG": {
            "title": "Врикшасана (Дарак позасы)",
            "technique": (
                "1. Тадасана позасынан баштаңыз. Салмакты сол бутка өткөрүңүз.\n"
                "2. Оң тизени бүгүп, оң буттун таманын сол буттун ички санына же балтырына коюңуз (тизеге тийгизбеңиз).\n"
                "3. Тирөөч бутту килемчеге бекем басып, көтөрүлгөн бутту санга түртүп, өз ара каршылыкты түзүңүз.\n"
                "4. Туруктуу караш чекитин табып, тең салмакты сактаңыз. Колдорду көкүрөк алдында намаскар абалында кармоого же жогору көтөрүүгө болот.\n"
                "5. Жамбашты бир деңгээлде сактап, омуртканы узартыңыз. 5–10 дем алуу кармап, андан кийин тарапты алмаштырыңыз."
            ),
            "benefits": (
                "• Тең салмакты жана көңүл топтоону жакшыртат.\n"
                "• Билектерди жана буттарды чыңдайт, жамбашты ачат.\n"
                "• Акыл-эстин туруктуулугун жана ички тең салмакты өнүктүрөт."
            ),
            "contra": "Бутту тизе муунуна койбоңуз. Эгер тең салмак туруксуз болсо, дубалды тирөөч катары колдонуңуз.",
            "image": "assets/vriksasana.png"
        }
    },

    "virabhadrasana i": {
        "EN": {
            "title": "Virabhadrasana I (Warrior I)",
            "technique": (
                "1. From standing, step the right foot forward about 3–4 feet. Left foot turned slightly out.\n"
                "2. Bend the front (right) knee so it is over the ankle (approx. 90°), back leg straight and strong.\n"
                "3. Square the hips to face the front leg; tuck tailbone slightly to protect the lower back.\n"
                "4. Reach arms overhead, palms touching or shoulder-width apart. Lift through the chest, soften the shoulders.\n"
                "5. Hold for 5–8 breaths, then step back and repeat on the other side."
            ),
            "benefits": (
                "• Strengthens legs, ankles, and glutes.\n"
                "• Opens chest and stretches hip flexors (back leg).\n"
                "• Builds stamina and stability."
            ),
            "contra": "Avoid if you have recent knee or hip injuries; modify depth of the knee bend.",
            "image": "assets/virabhadrasana_I.png"
        },
        "RU": {
            "title": "Вирабхадрасана I (Поза Воина I)",
            "technique": (
                "1. Из положения стоя сделайте шаг правой ногой вперёд на 3–4 фута (90–120 см). Левую стопу слегка разверните наружу.\n"
                "2. Согните переднее (правое) колено так, чтобы оно находилось над голеностопом (примерно 90°), задняя нога прямая и сильная.\n"
                "3. Разверните таз в сторону передней ноги; слегка подкрутите копчик, чтобы защитить поясницу.\n"
                "4. Поднимите руки вверх, ладони вместе или на ширине плеч. Раскрывайте грудную клетку, расслабляя плечи.\n"
                "5. Удерживайте позу 5–8 дыханий, затем сделайте шаг назад и повторите на другую сторону."
            ),
            "benefits": (
                "• Укрепляет ноги, голеностопы и ягодицы.\n"
                "• Раскрывает грудную клетку и растягивает сгибатели бедра (задняя нога).\n"
                "• Развивает выносливость и устойчивость."
            ),
            "contra": "Избегайте выполнения при недавних травмах коленей или тазобедренных суставов; уменьшайте глубину сгибания колена при необходимости.",
            "image": "assets/virabhadrasana_I.png"
        },
        "KG": {
            "title": "Вирабхадрасана I (Жоокер I)",
            "technique": (
                "1. Тик туруп, оң бутту алдыга 3–4 фут (90–120 см) кадам таштаңыз. Сол бутту бир аз сыртка буруңуз.\n"
                "2. Алдыңкы (оң) тизени билектин үстүнө чейин бүгүңүз (болжол менен 90°), арткы бут түз жана күчтүү болсун.\n"
                "3. Жамбашты алдыңкы бут тарапка түздөп, белди коргоо үчүн куймулчакты (tailbone) бир аз ичке тартыңыз.\n"
                "4. Колдорду жогору көтөрүңүз, алакандарды тийгизип же ийиндин туурасында кармаңыз. Көкүрөктү көтөрүп, ийиндерди бош кармаңыз.\n"
                "5. 5–8 дем алуу кармап, андан кийин артка кадам таштап, экинчи тарапка кайталаңыз."
            ),
            "benefits": (
                "• Буттарды, билектерди жана жамбаш булчуңдарын чыңдайт.\n"
                "• Көкүрөктү ачат жана арткы буттун жамбаш бүктөгүч булчуңдарын сунат.\n"
                "• Туруктуулукту жана чыдамкайлыкты өнүктүрөт."
            ),
            "contra": "Тизе же жамбаш жакында жаракат алган болсо, аткарбаңыз; тизени бүккөн тереңдикти азайтуу менен жеңилдетилген вариантты колдонуңуз.",
            "image": "assets/virabhadrasana_I.png"
        }
    },

    "virabhadrasana ii": {
        "EN": {
            "title": "Virabhadrasana II (Warrior II)",
            "technique": (
                "1. From standing, take a wide stance and turn the right foot out 90°, left foot slightly in.\n"
                "2. Bend the right knee so it is over the ankle (approx. 90°), keep the left leg straight and strong.\n"
                "3. Stretch the arms out to the sides at shoulder height, palms down, gaze over the front hand.\n"
                "4. Keep the torso upright (do not lean forward), sink into the front hip while keeping the back leg active.\n"
                "5. Hold 5–8 breaths, then repeat on the opposite side."
            ),
            "benefits": (
                "• Strengthens legs and opens the hips and chest.\n"
                "• Improves stamina and balance.\n"
                "• Increases hip flexibility."
            ),
            "contra": "Avoid deep twists of the neck; keep gaze soft if you have neck problems.",
            "image": "assets/virabhadrasana_II.png"
        },
        "RU": {
            "title": "Вирабхадрасана II (Поза Воина II)",
            "technique": (
                "1. Широко расставьте ноги, поверните правую стопу на 90°, левую — слегка внутрь.\n"
                "2. Согните правое колено до угла около 90°, левая нога прямая и активна.\n"
                "3. Вытяните руки в стороны на уровне плеч, ладони вниз, взгляд направлен над пальцами передней руки.\n"
                "4. Держите торс вертикально, не наклоняйтесь вперед; опускайтесь в переднем тазу, удерживая заднюю ногу напряжённой.\n"
                "5. Удерживайте 5–8 дыханий, затем повторите в другую сторону."
            ),
            "benefits": (
                "• Укрепляет ноги и раскрывает бедра и грудную клетку.\n"
                "• Повышает выносливость и устойчивость.\n"
                "• Улучшает гибкость бедер."
            ),
            "contra": "Не совершайте резких поворотов шеи; при проблемах с шеей взгляд держите мягким.",
            "image": "assets/virabhadrasana_II.png"
        },
        "KG": {
            "title": "Вирабхадрасана II (Жоокер II)",
            "technique": (
                "1. Буттарыңызды кенен ачып, оң буттун кетменин 90 градуска, ал эми сол бутту бир аз ичкери буруңуз.\n"
                "2. Оң тизеңизди болжол менен 90 градус бурчка чейин бүгүңүз, сол бут түз жана чың (активдүү) бойдон калсын.\n"
                "3. Колдоруңузду ийин деңгээлинде эки жакты карай сунуңуз, алакандарды ылдый каратып, көз карашты алдынкы колдун манжаларынан ашыра караңыз.\n"
                "4. Денени түз (вертикалдуу) кармаңыз, алдыга эңкейбеңиз; арткы бутту чыңалган абалда кармап, жамбаш менен ылдый түшүңүз.\n"
                "5. 5–8 жолу дем алганча ушул абалда туруңуз, андан кийин экинчи тарапка кайталаңыз."
            ),
            "benefits": (
                "• Буттарды бекемдейт, жамбашты жана көкүрөк клеткасын ачат.\n"
                "• Чыдамкайлыкты жана туруктуулукту жогорулатат.\n"
                "• Жамбаштын ийкемдүүлүгүн жакшыртат."
            ),
            "contra": "Мойнуңузду кескин бурбаңыз; эгер моюн жагынан көйгөй болсо, көз карашты жумшак кармаңыз.",
            "image": "assets/virabhadrasana_II.png"
        }
    },

    "adho mukha svanasana": {
        "EN": {
            "title": "Adho Mukha Svanasana (Downward-Facing Dog)",
            "technique": (
                "1. Start on hands and knees (tabletop). Place hands shoulder-width and knees hip-width.\n"
                "2. Spread fingers, press into palms, tuck toes and lift hips up and back into an inverted V shape.\n"
                "3. Keep the back long, encourage the heels toward the floor (knees may be slightly bent if hamstrings are tight).\n"
                "4. Relax the neck, let the head hang between the arms, breathe deeply for 5–8 breaths.\n"
                "5. To come out, lower knees to the mat or step forward to a standing position."
            ),
            "benefits": (
                "• Stretches the shoulders, hamstrings and calves.\n"
                "• Strengthens the arms and legs.\n"
                "• Calms the mind and relieves stress."
            ),
            "contra": "Avoid if you have severe wrist injury or uncontrolled high blood pressure; bend knees to reduce hamstring strain.",
            "image": "assets/adho_mukha_svanasana.png"
        },
        "RU": {
            "title": "Адхо Мукха Шванасана (Собака мордой вниз)",
            "technique": (
                "1. Встаньте на четвереньки: ладони под плечами, колени под бёдрами.\n"
                "2. Расставьте пальцы рук, упритесь в ладони, поднимите таз вверх и назад в форму перевёрнутой буквы V.\n"
                "3. Держите спину длинной, тяните пятки к полу (можно слегка согнуть колени при натянутых подколенных сухожилиях).\n"
                "4. Расслабьте шею, пусть голова висит между рук; дышите 5–8 дыханий.\n"
                "5. Чтобы выйти из асаны — опустите колени или шагните вперёд в стоячее положение."
            ),
            "benefits": (
                "• Растягивает плечи, подколенные сухожилия и икры.\n"
                "• Укрепляет руки и ноги.\n"
                "• Успокаивает ум и снимает стресс."
            ),
            "contra": "Избегать при серьёзных травмах запястья; при болях в спине согнуть колени.",
            "image": "assets/adho_mukha_svanasana.png"
        },
        "KG": {
            "title": "Адхо Мукха Шванасана (Төмөн караган ит)",
            "technique": (
                "1. Төрт аяктап туруңуз: алакандар ийиндердин астында, тизелер жамбаш сөөгүнүн астында болсун.\n"
                "2. Манжаларды кенен жайып, алаканга таянып, жамбашты өйдө жана артка көтөрүп, тескери 'V' тамгасынын формасын жасаңыз.\n"
                "3. Арканы түз кармап, согончокторду полго тартыңыз (эгер тизе тарамыштары тартылып жатса, тизени бир аз бүгүүгө болот).\n"
                "4. Моюнду бош коюңуз, баш колдордун ортосунда эркин турсун; 5–8 жолу дем алыңыз.\n"
                "5. Көнүгүүдөн чыгуу үчүн — тизени полго түшүрүңүз же алдыга кадам таштап, тик туруңуз."
            ),
            "benefits": (
                "• Ийиндерди, тизенин артындагы тарамыштарды жана балтырларды чоёт.\n"
                "• Колдорду жана буттарды бекемдейт.\n"
                "• Акыл-эсти тынчтандырат жана стресстен арылтат."
            ),
            "contra": "Билектин олуттуу жаракаты бар болсо, жасоодон алыс болуңуз; эгер бел ооруса, тизелерди бүгүңүз.",
            "image": "assets/adho_mukha_svanasana.png"
        }
    },

    "phalakasana": {
        "EN": {
            "title": "Phalakasana (Plank Pose)",
            "technique": (
                "1. Start in the top of a push-up position: hands under shoulders, legs long behind you.\n"
                "2. Align shoulders over wrists, draw the navel toward the spine and keep the body in a straight line from head to heels.\n"
                "3. Avoid dropping the hips (sagging) or lifting the hips too high. Keep gaze slightly forward.\n"
                "4. Hold 20–60 seconds depending on strength; breathe steadily."
            ),
            "benefits": (
                "• Strengthens core, shoulders, arms, and spine.\n"
                "• Improves posture and stability.\n"
                "• Prepares the body for arm balances."
            ),
            "contra": "Avoid if you have severe wrist or shoulder pain; modify on forearms if necessary.",
            "image": "assets/phalakasana.png"
        },
        "RU": {
            "title": "Пхалакасана (Планка)",
            "technique": (
                "1. Примите положение как для отжиманий: ладони под плечами, ноги вытянуты назад.\n"
                "2. Ладони под плечами, подтяните живот к позвоночнику, тело — в одну линию от макушки до пяток.\n"
                "3. Не прогибайтесь в пояснице и не поднимайте таз слишком высоко. Смотрите немного вперед.\n"
                "4. Удерживайте 20–60 секунд в зависимости от силы, дышите ровно."
            ),
            "benefits": (
                "• Укрепляет корпус, плечи и руки.\n"
                "• Улучшает осанку и устойчивость.\n"
                "• Подготавливает к балансовым позам."
            ),
            "contra": "Не делайте при сильной боли в запястьях или плечах; можно выполнять на предплечьях.",
            "image": "assets/phalakasana.png"
        },
        "KG": {
            "title": "Пхалакасана (Планка)",
            "technique": (
                "1. Колду жана бутту түз узатып, колдорду ийиндин астына коюңуз (push-up позициясы сыяктуу).\n"
                "2. Ийиндерди алакандарга туура келтирип, ичти ичке тартып, денени баштан бүтөнгө чейин түз кармаңыз.\n"
                "3. Жамбашты ылдый түшүрбөңүз жана жогору да көтөрбөңүз. Көзүңүздү алдыга каратыңыз.\n"
                "4. 20–60 секунда кармаңыз (кубатыңызга жараша)."
            ),
            "benefits": (
                "• Кор, ийин жана кол булчугун бекемдейт.\n"
                "• Осанканы жакшыртат.\n"
                "• Баланска даярдайт."
            ),
            "contra": "Билек же ийин оорулары барларда этият болуңуз; зарыл болсо предплечьяга отуңуз.",
            "image": "assets/phalakasana.png"
        }
    },

    "padmasana": {
        "EN": {
            "title": "Padmasana (Lotus Pose)",
            "technique": (
                "1. Sit with legs extended. Bend the right knee and place the right foot on the left thigh, then bend the left knee and place the left foot on the right thigh.\n"
                "2. Keep the spine straight, shoulders relaxed, and hands on the knees or in a mudra.\n"
                "3. Sit evenly on the sitting bones. Keep breathing calmly for meditation or pranayama."
            ),
            "benefits": (
                "• Calms the mind and supports meditation.\n"
                "• Stretches the ankles and knees (practice gently).\n"
                "• Improves hip mobility over time."
            ),
            "contra": "Avoid if you have acute knee or ankle injuries; use simpler seated postures if hips are tight.",
            "image": "assets/padmasana.png"
        },
        "RU": {
            "title": "Падмасана (Поза Лотоса)",
            "technique": (
                "1. Сядьте, вытянув ноги. Согните правое колено и положите стопу на левое бедро, затем левое колено и стопу на правое бедро.\n"
                "2. Держите позвоночник ровно, плечи расслаблены, руки на коленях или в мудре.\n"
                "3. Сидите ровно на седалищных костях. Дышите спокойно для медитации."
            ),
            "benefits": (
                "• Успокаивает ум, подходит для медитации.\n"
                "• Растягивает лодыжки и колени (выполнять аккуратно).\n"
                "• Со временем улучшает подвижность тазобедренных суставов."
            ),
            "contra": "Не занимайтесь лотосом при острых травмах колена или лодыжки; замените на простую сидячую позу.",
            "image": "assets/padmasana.png"
        },
        "KG": {
            "title": "Падмасана (Лотос позасы)",
            "technique": (
                "1. Отургучта бутту узатып отуруңуз. Оң тизени бүгүп, оң буттун таманын сол санга коюңуз, андан соң сол бутту оң санга коюңуз.\n"
                "2. Омуртка түз болсун, ийиндер бош, колдор тизеде же мудрада болсун.\n"
                "3. Отуруу учурунда тең болуңуз, тыныгуу менен дем алыңыз."
            ),
            "benefits": (
                "• Оюн тынчтандырат жана медитацияга ылайыктуу.\n"
                "• Тамандарды жана тизелерди созот (этият болуңуз).\n"
                "• Жамбаштын кыймылдуулугун жакшыртат."
            ),
            "contra": "Тизе же согончок жаракаттары барда жасабоо зарыл; жөнөкөй отурууга алмаштырыңыз.",
            "image": "assets/padmasana.png"
        }
    },

    "bhujangasana": {
        "EN": {
            "title": "Bhujangasana (Cobra Pose)",
            "technique": (
                "1. Lie prone with legs together and tops of the feet on the mat.\n"
                "2. Place hands under the shoulders, elbows close to the ribs.\n"
                "3. Inhale and lift the chest using the back muscles; keep a slight bend in the elbows and avoid pushing with the hands excessively.\n"
                "4. Keep the pelvis grounded and draw the shoulders away from the ears. Hold 3–6 breaths."
            ),
            "benefits": (
                "• Strengthens the spine and opens the chest.\n"
                "• Stimulates abdominal organs and improves posture.\n"
                "• Can help relieve mild back stiffness."
            ),
            "contra": "Avoid deep backbends if you have recent spinal injuries or severe lower-back pain.",
            "image": "assets/bhujangasana.png"
        },
        "RU": {
            "title": "Бхужангасана (Поза Кобры)",
            "technique": (
                "1. Лягте на живот, ноги вместе, ступни упираются на коврик.\n"
                "2. Положите ладони под плечи, локти прижаты к корпусу.\n"
                "3. На вдохе поднимите грудную клетку, работая мышцами спины; локти слегка согнуты, не отталкивайтесь слишком сильно руками.\n"
                "4. Бедра остаются на полу, плечи отведены от ушей. Удерживайте 3–6 вдохов."
            ),
            "benefits": (
                "• Укрепляет позвоночник и раскрывает грудь.\n"
                "• Стимулирует органы брюшной полости и улучшает осанку.\n"
                "• Помогает снять лёгкую скованность в спине."
            ),
            "contra": "Избегайте глубоких прогибов при острых травмах позвоночника или сильной боли в пояснице.",
            "image": "assets/bhujangasana.png"
        },
        "KG": {
            "title": "Бхужангасана (Кобра позасы)",
            "technique": (
                "1. Карынга жатып, буттарды бириктирип, буттун үстүн жерге тийгизиңиз.\n"
                "2. Колдорду ийиндин астына коюп, чыканактарды денеге жакын сактаңыз.\n"
                "3. Дем алып жатканда көкүрөктү омуртка булчуңдары менен көтөрүңүз; чыканактарды ашыкча түртпөңүз.\n"
                "4. Жамбаш жерде калсын, ийиндерди кулакка жакындатпаңыз. 3–6 дем кармаңыз."
            ),
            "benefits": (
                "• Омуртканы бекемдейт жана көкүрөктү ачат.\n"
                "• Ашказан жана ич органдарын стимулдайт.\n"
                "• Белдин жеңил катуулугун басат."
            ),
            "contra": "Омуртканын же белдин жаңы жаракаттары барда жасабоо керек.",
            "image": "assets/bhujangasana.png"
        }
    },

    "urdhva mukha svanasana": {
        "EN": {
            "title": "Urdhva Mukha Svanasana (Upward-Facing Dog)",
            "technique": (
                "1. Lie prone, hands beside the lower ribs.\n"
                "2. Press into the hands and the tops of the feet, lifting the chest and thighs off the floor so only the hands and tops of the feet touch.\n"
                "3. Open the chest, roll the shoulders back, and keep the arms straight without locking the elbows.\n"
                "4. Gaze forward or slightly upward. Hold 2–5 breaths and then lower with control."
            ),
            "benefits": (
                "• Strengthens the spine and arms.\n"
                "• Opens the chest and improves posture.\n"
                "• Counterposes forward folds and desk posture."
            ),
            "contra": "Avoid if you have severe back issues or wrist problems; use Cobra (Bhujangasana) as a gentler alternative.",
            "image": "assets/urdhva_mukha_svanasana.png"
        },
        "RU": {
            "title": "Урдхва Мукха Шванасана (Собака мордой вверх)",
            "technique": (
                "1. Лягте на живот, ладони рядом с грудной клеткой.\n"
                "2. Упритесь в ладони и поднимите грудь и бёдра, чтобы касались только ладони и верх стоп.\n"
                "3. Раскройте грудную клетку, сведите лопатки, держите локти прямыми, но не запирайте их.\n"
                "4. Смотрите прямо или слегка вверх. Удерживайте 2–5 дыханий, затем опуститесь плавно."
            ),
            "benefits": (
                "• Укрепляет позвоночник и руки.\n"
                "• Открывает грудь и улучшает осанку.\n"
                "• Является контрпозой к наклонам вперёд."
            ),
            "contra": "Не выполнять при серьёзных проблемах с поясницей; при болях замените на кобру.",
            "image": "assets/urdhva_mukha_svanasana.png"
        },
        "KG": {
            "title": "Урдхва Мукха Шванасана (Өйдө караган ит)",
            "technique": (
                "1. Көмкөрөдө жатып, колдорду көкүрөк жанына коюңуз.\n"
                "2. Алаканга басып, көкүрөктү жана санды жерден көтөрүп, кол жана буттун үстү гана тийсин.\n"
                "3. Көкүрөктү ачып, ийиндерди артка жылдырып, лактын тузун ачпай туруңуз.\n"
                "4. Аз гана өйдө караңыз. 2–5 дем кармаңыз, андан кийин жай түшүңүз."
            ),
            "benefits": (
                "• Омуртканы жана колду күчтүү кылат.\n"
                "• Көкүрөктү ачат жана осанканы жакшыртат.\n"
                "• Алдыңкы ийилүүлөргө каршы иштейт."
            ),
            "contra": "Бел же бүртүк оорулары бар болсо этият болуңуз; кобра менен алмаштырсаңыз болот.",
            "image": "assets/urdhva_mukha_svanasana.png"
        }
    },

    "salamba sarvangasana": {
        "EN": {
            "title": "Salamba Sarvangasana (Supported Shoulder Stand)",
            "technique": (
                "1. Lie on your back. Bend knees and lift legs overhead.\n"
                "2. Support your lower back with your hands and lift the hips, bringing the torso toward vertical (use blankets under shoulders if needed).\n"
                "3. Keep elbows close together, weight on the shoulders and upper arms; do not turn the head while in the pose.\n"
                "4. Keep legs active and pointed upward. Hold for as long as comfortable (start with 20–60 seconds).\n"
                "5. To come out, support hips, lower the legs slowly and roll down vertebra by vertebra."
            ),
            "benefits": (
                "• Improves circulation and calms the nervous system.\n"
                "• Stimulates the thyroid and digestive organs.\n"
                "• Strengthens shoulders and upper back."
            ),
            "contra": "Not recommended for people with neck injuries, high blood pressure, glaucoma or during menstruation.",
            "image": "assets/salamba_sarvangasana.png"
        },
        "RU": {
            "title": "Саламба Сарвангасана (Березка / Стойка на плечах)",
            "technique": (
                "1. Лягте на спину. Согните колени и поднимите ноги вверх.\n"
                "2. Поддерживая поясницу руками, поднимите таз и попытайтесь выстроить туловище вертикально (при необходимости подкладывайте одеяло под плечи).\n"
                "3. Держите локти близко друг к другу, вес на плечах и верхней части рук; не поворачивайте голову в позе.\n"
                "4. Держите ноги активными и направленными вверх. Начните с 20–60 секунд.\n"
                "5. Выходите, поддерживая таз, медленно опуская ноги и скручивая позвоночник."
            ),
            "benefits": (
                "• Улучшает кровообращение и успокаивает нервную систему.\n"
                "• Стимулирует щитовидку и органы пищеварения.\n"
                "• Укрепляет плечи и верхнюю часть спины."
            ),
            "contra": "Противопоказано при травмах шеи, высоком давлении, глаукоме и во время менструации.",
            "image": "assets/salamba_sarvangasana.png"
        },
        "KG": {
            "title": "Саламба Сарвангасана (Ийинге таянган / Березка)",
            "technique": (
                "1. Жерге жатыңыз. Тизелерди бүкүп, буттарды үстүңкү жакка көтөрүңүз.\n"
                "2. Белди кол менен кармап, жамбашты көтөрүп, денени вертикалга жакындаткыла (иягча кездемени ийиндин арт жагына коюңуз).\n"
                "3. Чыканыктар (локт) бири-бирине жакын болсун, дене салмагы ийиндерде болсун; моюнду бурбоңуз.\n"
                "4. Буттарды түз жана активдуу кармаңыз. 20–60 секундадан баштаңыз.\n"
                "5. Чыгуу үчүн жамбашты кармап, бутту жай түшүрүңүз."
            ),
            "benefits": (
                "• Канды жакшыраак айдайт жана нерв системасын тынчтандырат.\n"
                "• Щитовидка жана ичеги-карынга пайдалуу.\n"
                "• Ийин жана үстү жакты бекемдейт."
            ),
            "contra": "Моюн жаракаттары, жогорку кан басымы, глаукома же менструация учурунда жасоого болбойт.",
            "image": "assets/salamba_sarvangasana.png"
        }
    }
}

# -----------------------
# 2. IDEAL ANGLES CONFIG
# -----------------------
IDEAL_ANGLES = {
    "tadasana": {"knee": (170, 180), "hip": (170, 180)},
    "vriksasana": {"support_knee": (170, 180), "bent_knee": (30, 120), "hip": (160, 180)},
    "virabhadrasana i": {"front_knee": (80, 105), "back_knee": (170, 180), "arms": (160, 180)},
    "virabhadrasana ii": {"front_knee": (80, 105), "back_knee": (160, 180), "arms": (85, 95)},
    "adho mukha svanasana": {
        "shoulder_open": (160, 190), 
        "legs": (165, 180),
        "heels": (60, 120) 
    },
    "phalakasana": {"hip_line": (170, 180), "arms": (160, 180)},
    "padmasana": {"spine": (170, 180)},
    "bhujangasana": {"elbow": (30, 100), "lumbar_extension": (30, 80)},
    "urdhva mukha svanasana": {"arms": (160, 180), "spine": (140, 190)},
    "salamba sarvangasana": {"full_line": (170, 180)}
}

# -----------------------
# 3. HELPER FUNCTIONS
# -----------------------
def _to_xy(pt):
    if pt is None: return None
    if hasattr(pt, "x"): return (float(pt.x), float(pt.y))
    return (float(pt[0]), float(pt[1]))

def _angle_deg(a, b, c):
    if a is None or b is None or c is None: return None
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-8: return None
    cosv = np.dot(ba, bc) / denom
    cosv = max(min(cosv, 1.0), -1.0)
    ang = math.degrees(math.acos(cosv))
    return float(ang)

def _dist(a, b):
    if a is None or b is None: return None
    return float(np.linalg.norm(np.array(a) - np.array(b)))

def _get_front_leg_index(angles):
    lk = angles.get("left_knee")
    rk = angles.get("right_knee")
    if lk is None or rk is None: return None
    return 'left' if lk < rk else 'right'

def _compare(measured, ideal_min, ideal_max):
    if measured is None: return "unknown"
    if ideal_min <= measured <= ideal_max: return "ok"
    return "low" if measured < ideal_min else "high"

# -----------------------
# 4. FEATURE EXTRACTION & ANGLES
# -----------------------
def extract_features_from_landmarks(landmarks):
    pts = []
    raw_coords = []
    for i in range(33):
        lm = landmarks[i]
        coords = (float(lm.x), float(lm.y), float(lm.z)) if hasattr(lm, "x") else (float(lm[0]), float(lm[1]), float(lm[2]))
        pts.append(coords)
        raw_coords.extend(coords)

    p2 = lambda idx: (pts[idx][0], pts[idx][1])
    l_sh, r_sh = p2(11), p2(12)
    l_wr, r_wr = p2(15), p2(16)
    l_hip, r_hip = p2(23), p2(24)
    l_ank, r_ank = p2(27), p2(28)

    mid_sh = ((l_sh[0]+r_sh[0])/2, (l_sh[1]+r_sh[1])/2)
    mid_hip = ((l_hip[0]+r_hip[0])/2, (l_hip[1]+r_hip[1])/2)
    scale = _dist(mid_sh, mid_hip) or 1.0
    if scale < 1e-6: scale = 1.0

    feats_geo = np.array([
        _dist(l_ank, r_ank) / scale,
        _dist(l_wr, r_wr) / scale,
        _dist(l_wr, l_sh) / scale,
        _dist(r_wr, r_sh) / scale,
        _angle_deg(p2(23), p2(25), p2(27)) or 0,
        _angle_deg(p2(24), p2(26), p2(28)) or 0,
        _angle_deg(p2(11), p2(23), p2(25)) or 0,
        _angle_deg(p2(12), p2(24), p2(26)) or 0,
        _angle_deg(p2(11), p2(13), p2(15)) or 0,
        _angle_deg(p2(12), p2(14), p2(16)) or 0,
        _angle_deg(p2(13), p2(11), p2(23)) or 0,
        _angle_deg(p2(14), p2(12), p2(24)) or 0 
    ], dtype=float)

    final_features = np.concatenate([feats_geo, np.array(raw_coords)])
    z_list = [p[2] for p in pts]
    return final_features.reshape(1, -1), z_list

def compute_relevant_angles(landmarks):
    p2 = lambda idx: _to_xy(landmarks[idx]) if hasattr(landmarks[idx], "x") else (landmarks[idx][0], landmarks[idx][1])
    angles = {}
    
    # Joints
    angles['left_knee'] = _angle_deg(p2(23), p2(25), p2(27))
    angles['right_knee'] = _angle_deg(p2(24), p2(26), p2(28))
    angles['left_hip'] = _angle_deg(p2(11), p2(23), p2(25))
    angles['right_hip'] = _angle_deg(p2(12), p2(24), p2(26))
    angles['left_elbow'] = _angle_deg(p2(11), p2(13), p2(15))
    angles['right_elbow'] = _angle_deg(p2(12), p2(14), p2(16))
    
    # Arm Elevation (Torso-Shoulder-Elbow)
    angles['left_shoulder'] = _angle_deg(p2(23), p2(11), p2(13)) 
    angles['right_shoulder'] = _angle_deg(p2(24), p2(12), p2(14))
    
    angles['torso_left'] = _angle_deg(p2(11), p2(23), p2(25)) 
    angles['left_shoulder_open'] = _angle_deg(p2(15), p2(11), p2(23))
    angles['right_shoulder_open'] = _angle_deg(p2(16), p2(12), p2(24))
    angles['left_ankle'] = _angle_deg(p2(25), p2(27), p2(31))
    angles['right_ankle'] = _angle_deg(p2(26), p2(28), p2(32))
    
    # === Torso Symmetry (Side Lengths) ===
    # Calculate distance from shoulder (11/12) to hip (23/24)
    dist_l = _dist(p2(11), p2(23))
    dist_r = _dist(p2(12), p2(24))
    
    # Ratio: > 1.0 means left side is longer (leaning right/crunching right)
    # < 1.0 means right side is longer (leaning left/crunching left)
    if dist_r and dist_r > 0:
        angles['torso_symmetry'] = dist_l / dist_r
    else:
        angles['torso_symmetry'] = 1.0

    for k in list(angles.keys()):
        if k != 'torso_symmetry': 
            angles[k] = None if angles[k] is None else round(float(angles[k]), 1)
            
    return angles

# -----------------------
# 5. FEEDBACK LOGIC
# -----------------------
FEEDBACK_TEXT = {
    "EN": {
        "front_knee_high": "Bend your front knee deeper (aim for 90°).", 
        "front_knee_low": "Don't bend your knee too much (keep it above ankle).",
        "back_knee_low": "Straighten the back/standing leg.",
        
        # --- Missing keys added back ---
        "spine_low": "Lift your hips slightly! Don't let your lower back sag.",
        "arms_low": "Lift and straighten your arms.",
        "arms_high": "Lower your arms to shoulder level.",
        "front_arm_high": "Lower your front arm slightly.",
        "back_arm_low": "Lift your back arm up.",
        
        "torso_long_left": "Drop your left shoulder down. Don't lean forward.",
        "torso_long_right": "Drop your right shoulder down. Keep your torso vertical.",
        
        "shoulder_low": "Push the floor away! Align arms with torso.",
        "heels_fix": "Lower your heels to the floor! If needed, bend knees slightly but keep heels grounded.",
        "heels_up": "Your heels are lifted! Press them firmly into the mat.",
        "ok": "Good alignment"
    },
    "RU": {
        "front_knee_high": "Согните переднее колено глубже (стремитесь к 90°).",
        "front_knee_low": "Не сгибайте колено слишком сильно.",
        "back_knee_low": "Выпрямите заднюю/опорную ногу.",
        
        # --- Вернули недостающие ключи ---
        "spine_low": "Поднимите таз выше! Не прогибайтесь в пояснице.",
        "arms_low": "Поднимите и выпрямите руки.",
        "arms_high": "Опустите руки до уровня плеч.",
        "front_arm_high": "Опустите переднюю руку чуть ниже.",
        "back_arm_low": "Поднимите заднюю руку выше.",
        
        "torso_long_left": "Опустите левое плечо. Не тянитесь корпусом вперед.",
        "torso_long_right": "Опустите правое плечо. Держите корпус вертикально.",
        
        "shoulder_low": "Толкните пол руками! Выстройте руки и спину в одну линию.",
        "heels_fix": "Опустите пятки на пол! Если тяжело — согните колени, но пятки должны касаться пола.",
        "heels_up": "Пятки оторваны! Прижмите их к коврику.",
        "ok": "Хорошее положение"
    },
    "KG": {
        "front_knee_high": "Алдыңкы тизени тереңирээк бүгүңүз (90°).",
        "front_knee_low": "Тизени ашыкча бүкпөңүз.",
        "back_knee_low": "Арткы/тирөөч бутту түздөңүз.",
        
        # --- Жетишпеген ачкычтар ---
        "spine_low": "Жамбашты бир аз көтөрүңүз! Белди ылдый түшүрбөңүз.",
        "arms_low": "Колдорду жогору көтөрүп, түздөңүз.",
        "arms_high": "Колдорду ылдый түшүрүңүз.",
        "front_arm_high": "Алдыңкы колду бир аз түшүрүңүз.",
        "back_arm_low": "Арткы колду бир аз көтөрүңүз.",
        
        "torso_long_left": "Сол ийинди түшүрүңүз. Денени алдыга созбоңуз.",
        "torso_long_right": "Оң ийинди түшүрүңүз. Денени түз кармаңыз.",
        
        "shoulder_low": "Полду колуңуз менен түртүңүз! Кол менен арканы бир сызыкка түздөңүз.",
        "heels_fix": "Согончокторду полго басыңыз! Керек болсо тизени бүгүңүз, бирок согончок полдо болсун.",
        "heels_up": "Согончоктор көтөрүлүп турат! Аларды басыңыз.",
        "ok": "Абалы жакшы"
    }
}


def generate_feedback(pose_name, angles, lang_code="EN"):
    key = pose_name.strip().lower()
    ideal = IDEAL_ANGLES.get(key)
    if not ideal: return []

    txt = FEEDBACK_TEXT.get(lang_code, FEEDBACK_TEXT["EN"])
    feedback = []

    # 1. WARRIOR I & II
    if key in ["virabhadrasana i", "virabhadrasana ii"]:
        # Determine front leg
        front_side = _get_front_leg_index(angles) # 'left' or 'right'
        
        # Check Knees
        if front_side:
            fk_angle = angles.get(f"{front_side}_knee")
            bk_angle = angles.get(f"{'right' if front_side=='left' else 'left'}_knee")
            
            ideal_range = ideal.get("front_knee", (80, 105))
            if fk_angle > ideal_range[1]: 
                feedback.append("👉 " + txt["front_knee_high"])
            elif fk_angle < ideal_range[0]:
                feedback.append("👉 " + txt["front_knee_low"])
                
            if bk_angle < ideal.get("back_knee", (170, 180))[0]:
                feedback.append("👉 " + txt["back_knee_low"])
        
        # === Check Torso Symmetry (Side Lengths) ===
        ratio = angles.get('torso_symmetry', 1.0)
        
        if ratio > 1.05:
            feedback.append("👉 " + txt["torso_long_left"])
        elif ratio < 0.95:
            feedback.append("👉 " + txt["torso_long_right"])

        # === Check Arms (Warrior II Strict) ===
        if key == "virabhadrasana ii" and front_side:
            front_sh_angle = angles.get(f"{front_side}_shoulder")
            back_sh_angle = angles.get(f"{'right' if front_side=='left' else 'left'}_shoulder")
            
            # STRICT thresholds for demo: 
            # If front arm > 92 (slightly above parallel) -> Lower it
            if front_sh_angle > 92:
                feedback.append("👉 " + txt["front_arm_high"])
            elif front_sh_angle < 80:
                feedback.append("👉 " + txt["arms_low"]) 
                
            # If back arm < 88 (slightly below parallel) -> Lift it
            if back_sh_angle < 88:
                feedback.append("👉 " + txt["back_arm_low"])
            elif back_sh_angle > 105:
                feedback.append("👉 " + txt["arms_high"])

    # 2. TREE
    elif key == "vriksasana":
        lk, rk = angles.get("left_knee"), angles.get("right_knee")
        standing_leg = max(lk, rk) if (lk and rk) else (lk or rk)
        if standing_leg and standing_leg < 160: feedback.append("👉 " + txt["back_knee_low"])

    # 3. PLANK
    elif key == "phalakasana":
        hip = angles.get("torso_left") 
        limit = ideal.get("hip_line", (170, 180))[0]
        if hip and hip < limit: feedback.append("👉 " + txt["spine_low"])

    # 4. COBRA
    elif key == "bhujangasana":
        elbow = angles.get("left_elbow") or angles.get("right_elbow")
        limit_max = ideal.get("elbow", (30, 100))[1]
        if elbow and elbow > limit_max: feedback.append("👉 " + txt["arms_high"])

    # 5. UPWARD DOG
    elif key == "urdhva mukha svanasana":
        elbow = angles.get("left_elbow") or angles.get("right_elbow")
        limit_min = ideal.get("arms", (160, 180))[0]
        if elbow and elbow < limit_min: feedback.append("👉 " + txt["arms_low"])

    # 6. DOWNWARD DOG
    elif key == "adho mukha svanasana":
        ls, rs = angles.get("left_shoulder_open"), angles.get("right_shoulder_open")
        current_shoulder = max(ls or 0, rs or 0)
        if current_shoulder < ideal.get("shoulder_open", (160, 190))[0]:
            feedback.append("👉 " + txt["shoulder_low"])

        la, ra = angles.get("left_ankle"), angles.get("right_ankle")
        current_ankle = max(la or 0, ra or 0)
        lk, rk = angles.get("left_knee"), angles.get("right_knee")
        current_knee = max(lk or 0, rk or 0)
        
        if current_ankle > 125 or current_knee < ideal.get("legs", (165, 180))[0]:
             feedback.append("👉 " + txt["heels_fix"])

    if not feedback: feedback.append("✅ " + txt["ok"])
    return feedback