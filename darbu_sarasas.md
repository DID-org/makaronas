
---

# Komandos GitHub užduotys (Issues) — Juodraštis

*Data: 2026-03-05*

---

## Apžvalga

Šias užduotis galima saugiai vykdyti lygiagrečiai su V5 (*Trickster Engine*) diegimu. Jos apima turinį, įrankius ir dokumentaciją — neliečiant DI (*AI*) sluoksnio ar *backend* variklio kodo.

**Failai, kuriuos komanda turėtų perskaityti pirmiausia:**

* `content/tasks/TEMPLATE/AUTHORING_GUIDE.md` — pilnas turinio kūrimo procesas (*workflow*)
* `content/tasks/TEMPLATE/task.json` — kasetės (*cartridge*) šablonas su vietos užpildais (*placeholders*)
* `content/tasks/task-clickbait-trap-001/task.json` — pavyzdinė hibridinė užduotis (baigta)
* `content/tasks/task-phantom-quote-001/task.json` — pavyzdinė `ai_driven` užduotis (baigta)
* `content/taxonomy.json` — žinomi trigeriai, technikos, terpės (*mediums*)
* `content/tasks/task.schema.json` — JSON schema skirta IDE validacijai

---

## 1 Užduotis: Pridėti `VideoBlock` prie užduočių schemų

**Tipas:** Kodas (nedidelis)
**Laiko įvertis:** 30 min.
**Failas:** `backend/tasks/schemas.py`

**Kas:**
Pridėkite `VideoBlock` klasę prie užduoties schemos, sekant lygiai tokiu pačiu principu kaip `ImageBlock`. Užregistruokite ją `KNOWN_BLOCK_TYPES` sąraše.

**Kodėl:**
Platformai reikia palaikyti video turinį užduotyse (pvz., DI sugeneruotų video palyginimo užduotims). Šiuo metu schemoje yra `ImageBlock`, `AudioBlock` ir `VideoTranscriptBlock`, bet nėra bloko patiems video failams.

**Pavyzdys, kuriuo sekame (nukopijuoti ir pritaikyti):**

```python
# Pažiūrėkite į ImageBlock (~71 eilutė) ir išlaikykite tą pačią struktūrą:
class VideoBlock(BaseModel):
    """Video content — AI-generated videos, news clips, social media videos.

    Accessibility: alt_text is required (Framework Principle 14).
    transcript provides text alternative for deaf/hard-of-hearing students.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    type: Literal["video"] = "video"
    src: str
    alt_text: str
    transcript: str | None = None
    duration_seconds: int | None = None

```

**Priėmimo kriterijai (Acceptance criteria):**

* [ ] `VideoBlock` klasė sukurta `backend/tasks/schemas.py`
* [ ] Užregistruota `KNOWN_BLOCK_TYPES` žodyne
* [ ] `alt_text` yra privalomas (14-as karkaso principas — prieinamumas)
* [ ] `transcript` yra nebūtinas (ne visi video turi transkripcijas kūrimo metu)
* [ ] Esami testai praeina (paleiskite `python -m pytest backend/tests/ -v`)
* [ ] Pridėtas testas schemos testuose, kuris patvirtina, kad kasetė su `VideoBlock` užsikrauna teisingai

---

## 2 Užduotis: Sukurti kasečių validavimo CLI įrankį

**Tipas:** Kodas (įrankiai)
**Laiko įvertis:** 2-3 val.
**Failas:** Naujas failas — `scripts/validate_cartridge.py`

**Kas:**
Komandinės eilutės (*CLI*) skriptas, kuris validuoja užduoties kasetę per visą krovimo ciklą (*loader pipeline*) ir pateikia klaidas žmonėms suprantamu formatu. Turinio autoriai turėtų jį paleisti prieš *commitindami*.

**Kodėl:**
Šiuo metu validacija vyksta tik paleidžiant serverį (*loader*) arba testų metu. Turinio autoriams reikia greito būdo patikrinti savo darbą nepaleidžiant viso serverio ar testų paketo.

**Naudojimas:**

```bash
# Validuoti vieną kasetę
python scripts/validate_cartridge.py content/tasks/my-new-task-001

# Validuoti visas kasetes
python scripts/validate_cartridge.py content/tasks/

```

**Įgyvendinimo planas:**

* Importuoti `TaskLoader` iš `backend/tasks/loader.py`
* Importuoti taksonomiją iš `content/taxonomy.json`
* Kviesti `loader.load_task()` su taksonomijos kontekstu
* Pagauti ir suformatuoti visas validacijos klaidas (`LoadError`, `ValidationError`, įspėjimus)
* Išvesti aiškias, pataisomas žinutes: „Fazė 'intro' nurodo bloką 'main-image', bet bloko su tokiu ID nėra `presentation_blocks` sąraše“
* Išėjimo kodas (*exit code*): 0 = švaru, 1 = klaidos, 2 = tik įspėjimai

**Priėmimo kriterijai:**

* [ ] Skriptas pasileidžia iš projekto šaknies (*root*)
* [ ] Atpažįsta ir suformatuoja schemos validacijos klaidas
* [ ] Atpažįsta ir suformatuoja *loaderio* verslo logikos klaidas (kelio neatitikimai (*path mismatch*), našlaitinės fazės, trūkstami failai)
* [ ] Praneša apie taksonomijos įspėjimus (nežinomi `trigger`/`technique`/`medium` kintamieji)
* [ ] Aiški klaidų informacija, pasakanti autoriui, ką taisyti (o ne *stack trace'as*)
* [ ] Veikia ir su viena kasete, ir su visu katalogu
* [ ] Užbaigia darbą su atitinkamais *exit* kodais

---

## 3 Užduotis: Naujos užduoties juodraštis — Manipuliacinis dialogas (Klaidinanti statistika)

**Tipas:** Turinys
**Laiko įvertis:** 3-4 val.
**Failai:** Naujas katalogas `content/tasks/task-misleading-stats-001/`

**Kas:**
Sukurti naują hibridinę užduotį, kurioje mokiniui rodomas socialinių tinklų įrašas, cituojantis tikrą tyrimą, tačiau statistika pateikiama klaidinančiai (procentai be bazinių skaičių, koreliacija pateikiama kaip priežastingumas, specialiai atrinktas laiko tarpas).

**Archetipas:** Adversarial Dialogue (seka `task-clickbait-trap-001` pavyzdžiu)
**Personažo režimas:** `presenting`
**Terpė (Medium):** `social_post`
**Technika:** `cherry_picking`
**Trigeris:** `authority`

**Turinio reikalavimai:**

* Socialinio tinklo įrašo blokas (`social_post` tipo) su klaidinančia statistika apie paaugliams aktualią temą (laikas prie ekrano, socialiniai tinklai, pasiekimai mokykloje — tema turi būti visada aktuali).
* Šaltinio duomenų blokas (`text` tipo), rodantis tikrus tyrimo duomenis, kurie atskleidžia manipuliaciją.
* Statinė `intro` fazė su mygtukais (dalintis / kvestionuoti / tirti).
* DI vertinimo fazė (min 2, max 6 apsikeitimai žinutėmis).
* Trys atskleidimo (*reveal*) fazės (laimėjimas / dalinė sėkmė / laikas baigėsi) su lietuvišku tekstu, paaiškinančiu konkretų statistinį triuką.
* Vertinimo kontraktas: bent 2 šablonai `patterns_embedded` sąraše, bent 1 privalomas (*mandatory*) *checklist'o* punktas.

**Kokybės reikalavimai (Checklist):**

* [ ] Visas tekstas lietuvių kalba
* [ ] Jokių tikrų institucijų, žmonių ar dabartinių įvykių (4 principas — *Evergreen*)
* [ ] `task_id` atitinka katalogo pavadinimą
* [ ] `status: "draft"`
* [ ] Statistika atrodo realistiškai, bet yra išgalvota
* [ ] Manipuliacija yra konkreti ir atpažįstama (ne šiaip miglotas „klaidinimas“ — o tiksliai pasakyta, koks skaičius ir kodėl yra neteisingas)
* [ ] *Reveal* tekstas paaiškina pačią techniką, o ne tik sako „tu klydai“
* [ ] Praeina `python scripts/validate_cartridge.py` testą (arba *loader* testą)

**Nuoroda:** Pradžiai nukopijuokite `content/tasks/TEMPLATE/`. Išnagrinėkite `task-clickbait-trap-001` dėl hibridinio modelio struktūros.

---

## 4 Užduotis: Naujos užduoties juodraštis — Manipuliacinis dialogas (Emocionali antraštė)

**Tipas:** Turinys
**Laiko įvertis:** 3-4 val.
**Failai:** Naujas katalogas `content/tasks/task-emotional-headline-001/`

**Kas:**
Sukurti naują hibridinę užduotį, kurioje mokinys mato naujienų straipsnį su emociškai užkrauta antrašte, kuri iškreipia tikrąjį straipsnio turinį. Pats straipsnis yra subalansuotas, bet antraštė parenka patį provokuojantį kampą.

**Archetipas:** Adversarial Dialogue (seka `task-clickbait-trap-001` pavyzdžiu)
**Personažo režimas:** `presenting`
**Terpė (Medium):** `article`
**Technika:** `headline_manipulation`
**Trigeris:** `injustice`

**Turinio reikalavimai:**

* Teksto blokas su pilnu straipsniu (subalansuotu, faktuotu).
* Antraštė, kuri jį iškreipia (emocionali, provokuojanti).
* Kontrastas turi būti aiškus skaitant atidžiai, bet lengvai praleidžiamas „praskrolinant“ (būtent taip paaugliai ir vartoja turinį).
* Statinė `intro` fazė, DI vertinimo fazė, trys atskleidimo (*reveal*) fazės.
* Vertinimo kontraktas: šablonai, apimantys antraštės ir teksto neatitikimą, emocinį rėminimą (*framing*), niuansų praleidimą.

**Kokybės reikalavimai (Checklist):**

* Tokie patys kaip 3 Užduotyje.

**Nuoroda:** Išnagrinėkite `task-clickbait-trap-001` — struktūra panaši, bet skiriasi manipuliacijos technika.

---

## 5 Užduotis: Naujos užduoties juodraštis — Tyrimas (Šaltinių atsekimas)

**Tipas:** Turinys
**Laiko įvertis:** 4-5 val.
**Failai:** Naujas katalogas `content/tasks/task-source-trace-001/`

**Kas:**
Sukurti naują hibridinę tyrimo (*investigation*) užduotį, kurioje mokinys atseka teiginį per kelis šaltinius ir atranda, kad kiekvienas „šaltinis“ remiasi ankstesniuoju ratu — niekas iš tiesų nepatikrino originalaus fakto.

**Archetipas:** Investigation (seka `task-follow-money-001` pavyzdžiu)
**Personažo režimas:** `narrator`
**Terpė (Medium):** `investigation`
**Technika:** `source_weaponization`
**Trigeris:** `authority`

**Turinio reikalavimai:**

* Keli `search_result` blokai, formuojantys tyrimo medį.
* Pėdsakas turi atrodyti kaip tikras tyrimas — kiekvienas rezultatas atskirai atrodo patikimas.
* Bent 2 raktiniai atradimai (*key findings*) ir 2 aklavietės (*dead ends*).
* `starting_queries`, kurie duoda mokiniui aiškius įėjimo taškus.
* Žiedinio citavimo (*circular reference*) raštas turi būti atrandamas, bet ne akivaizdus.
* DI vertinimo fazė, kurioje *Tricksteris* (kaip pasakotojas) veda mokinį per įrodymus.
* *Reveal* tekstas, paaiškinantis, kaip žiedinis citavimas veikia realioje medijoje.

**Tai pati sudėtingiausia turinio užduotis.** Prieš pradedant, atidžiai išstudijuokite `task-follow-money-001`.

**Kokybės reikalavimai (Checklist):**

* Tokie patys kaip 3-4 Užduotyse, plius:
* [ ] Tyrimo medyje nėra „niekur nevedančių kilpų” (kiekvienas kelias arba pasiekia raktinį atradimą, arba aiškiai pažymėtą aklavietę)
* [ ] `search_result` blokai turi realistiškus `query`, `title` ir `snippet` laukus
* [ ] `child_queries` teisingai veda tyrimą į priekį

---

## 6 Užduotis: Surinkti / Sukurti vaizdo išteklius (Assets) vizualinės manipuliacijos užduotims

**Tipas:** Turinys (medija)
**Laiko įvertis:** Tęstinis
**Failai:** `content/tasks/task-misleading-frame-001/assets/` ir naujų užduočių *assets* katalogai

**Kas:**
Sukurti arba surasti vaizdo išteklius užduotims, susijusioms su vizualine manipuliacija. Esamas `task-misleading-frame-001` turi tik laikinus paveiksliukus (`misleading.png`, `context.png`). Mums reikia:

1. **task-misleading-frame-001 užduočiai:** Dviejų tos pačios scenos nuotraukų iš skirtingų kampų / apkirpimų — viena pasakoja klaidinančią istoriją, kita rodo pilną kontekstą. Pavyzdys: apkirpta nuotrauka rodo „tuščią“ renginį, o pilna nuotrauka iš kito kampo rodo pilną salę.
2. **Ateities užduotims:** Sukurti nedidelę biblioteką, kurioje būtų:
* Klaidinantys grafikai (tikri duomenys, bet klaidinantis pateikimas — apkarpyta Y ašis, specialiai atrinktas laikotarpis)
* Apkirptos vs. pilno konteksto nuotraukos
* Išgalvotų socialinių tinklų įrašų ekrano nuotraukos (*screenshots*), stilizuotos priminti tikras platformas, bet jų nekopijuojančios (žr. 5 principą).



**Reikalavimai:**

* Visi vaizdai turi būti originalūs arba CC0/viešo naudojimo (*public domain*) — jokio autorinėmis teisėmis saugomo turinio.
* Jokių tikrų žmonių, tikrų prekių ženklų ar tikrų įvykių (4 principas — *Evergreen*).
* Kiekvienas vaizdas kasetėje privalo turėti `alt_text` (14 principas — prieinamumas).
* Vaizdai turi būti pakankamai aiškūs mokyklinio nešiojamojo kompiuterio ekrane (gera rezoliucija, geras kontrastas).
* Grafikai turi naudoti išgalvotus, bet įtikinamus duomenis.

**Priėmimo kriterijai:**

* [ ] `task-misleading-frame-001` turi produkcijos lygio nuotraukas, pakeičiančias laikinus paveiksliukus
* [ ] Paruošti bent 2 klaidinančių grafikų rinkinius (grafikas + šaltinio duomenys) ateities užduotims
* [ ] Paruoštos bent 2 „apkirpta vs. kontekstas“ nuotraukų poros ateities užduotims
* [ ] Visi failai yra `content/tasks/{task_id}/assets/` kataloguose

---

## 7 Užduotis: Išplėsti taksonomiją

**Tipas:** Turinys (duomenys)
**Laiko įvertis:** 1-2 val.
**Failas:** `content/taxonomy.json`

**Kas:**
Peržiūrėti dabartinę taksonomiją ir pasiūlyti papildymus, remiantis Lietuvos medijų erdve ir dažnais manipuliacijos modeliais, su kuriais susiduria paaugliai.

**Dabartinė taksonomija:**

* **Trigeriai (8):** urgency, belonging, injustice, authority, identity, fear, greed, cynicism
* **Technikos (10):** cherry_picking, fabrication, emotional_framing, wedge_driving, omission, false_authority, manufactured_deadline, headline_manipulation, source_weaponization, phantom_quote
* **Terpės / Mediums (9):** article, social_post, chat, investigation, meme, feed, audio, video_transcript, image

**Klausimai apmąstymui:**

* Ar yra manipuliacijos technikų, būdingų Lietuvos medijoms, kurių čia neįtraukėme? (pvz., `whataboutism`, `false_equivalence`, `appeal_to_tradition`)
* Ar yra trigerių, specifinių Lietuvos kontekstui? (pvz., susijusių su nacionaliniu identitetu, geopolitinėmis baimėmis)
* Ar yra terpių, kurias paaugliai naudoja, bet mes jų neturime? (pvz., `video`, `screenshot`, `voice_message`)
* Ar lietuviški rodymo pavadinimai (*display names*) taksonomijoje yra tikslūs ir natūralūs?

**Procesas:**

1. Peržiūrėti esamą `taxonomy.json`
2. Pasidomėti dažnais manipuliacijos modeliais Lietuvos medijose (naujienų portaluose, socialiniuose tinkluose, Telegram kanaluose)
3. Pasiūlyti papildymus kaip *PR'ą* (Pull Request) — pridėti naujus įrašus į `taxonomy.json` su lietuviškais *display names*
4. Aptarti su komanda prieš *merdžinant* — taksonomijos reikšmės rodomos vertinimo duomenyse

**Priėmimo kriterijai:**

* [ ] Bent 3-5 naujų technikų ar trigerių pasiūlymai su lietuviškais *display names*
* [ ] Kiekvienas papildymas turi 1 sakinio pagrindimą (kodėl tai aktualu Lietuvos paaugliams)
* [ ] Nėra esamų reikšmių dublikatų (tikrinti semantinį persidengimą, ne tik teksto atitikimą)
* [ ] Pridėtas `medium: "video"` (kadangi pridedame `VideoBlock` palaikymą)

---

## 8 Užduotis: Naujos užduoties juodraštis — Švarus patikrinimas (Patikimas straipsnis)

**Tipas:** Turinys
**Laiko įvertis:** 3-4 val.
**Failai:** Naujas katalogas `content/tasks/task-clean-check-001/`

**Kas:**
Sukurti pirmąją „švarią“ užduotį — straipsnį BE jokios manipuliacijos, kurį mokinys turi atpažinti kaip patikimą. Tai tikrina „klaidingai teigiamo“ instinktą: ne viskas yra apgavystė.

**Archetipas:** Clean Check
**Personažo režimas:** `presenting`
**Terpė (Medium):** `article`
**`is_clean`: `true**`

**Turinio reikalavimai:**

* Gerai parašytas, subalansuotas straipsnis paaugliams aktualia tema.
* Straipsnis turi būti pavyzdinė žurnalistika — subalansuota, pagrįsta šaltiniais, su niuansais.
* *Tricksteris* pateikia jį su tokiu pat pasitikėjimu, kaip ir manipuliuotą straipsnį — mokinys turi nuspręsti pats.
* `patterns_embedded` PRIVALO būti tuščias masyvas (`is_clean` + šablonai = kritinė *loaderio* klaida).
* Vertinimas (*evaluation*) apsiverčia: „Tricksteris laimi“ = mokinys neteisingai apkaltino švarų turinį (paranoja nugalėjo sveiką protą).
* *Reveal* tekstas paaiškina, kas padarė šį straipsnį patikimą ir kodėl mokinio įtarimai buvo nepagrįsti (arba pasveikina juos atpažinus gerą turinį).

**Tai pedagogiškai labai svarbu.** Be „švarių“ užduočių mes treniruojame paranoją, o ne sprendimų priėmimą. *Reveal* tekstas turi gerbti mokinį — „Geras sprendimas reiškia žinojimą, kada pasitikėti, o ne tik kada abejoti.“

**Kokybės reikalavimai (Checklist):**

* Tokie patys kaip 3-4 Užduotyse, plius:
* [ ] `is_clean: true`
* [ ] `patterns_embedded: []` (tuščias)
* [ ] Straipsnyje tikrai nėra jokios manipuliacijos (net ir subtilios)
* [ ] Vertinimo `pass_conditions` teisingai aprašo apverstus rezultatus
* [ ] *Reveal* tekstas apima „kalibravimo“ žinutę — pasitikėjimas geru turiniu taip pat yra įgūdis

**Pastaba:** DI (*AI*) promptas švarioms užduotims yra V5 variklio dalis (variklis turi palaikyti apverstą vertinimą). Komanda dabar tik parašo turinį; V5 sugeneruos promptą. Užduoties `status` liks `"draft"`, kol V5 bus baigtas.

---

## 9 Užduotis: Paruošti scenarijų aprašymus (*Briefs*) ateities užduotims

**Tipas:** Turinys (planavimas)
**Laiko įvertis:** 2-3 val.
**Failai:** Naujas failas `content/tasks/SCENARIO_BRIEFS.md`

**Kas:**
Parašyti trumpus scenarijų aprašymus (maždaug po pusę puslapio) 5-8 naujoms užduotims per skirtingus archetipus. Tai nėra pilnos kasetės — tai tik idėjos, kurias komanda vėliau galės paversti kasetėmis.

**Kiekvienas aprašymas turi apimti:**

* Užduoties pavadinimą ir archetipą (*adversarial dialogue, investigation, clean check, sensory trap, empathy flip*)
* Temą ir manipuliacijos techniką
* 2-3 sakinių scenarijaus aprašymą
* Kas daro šią užduotį įdomią/unikalią Lietuvos paaugliams
* Numatomas sunkumas (1-5)
* Kokių turinio blokų jai reikės (tekstas, nuotraukos, audio, video, *chat* žinutės)

**Tikslas:** Sukurti užduočių srautą (*pipeline*), kad komanda visada turėtų kitą užduotį paruoštą kūrimui. Siekite įvairovės per archetipus, technikas, terpes ir sunkumo lygius.

**Bandomosios versijos tikslas yra ~15-20 užduočių.** Mes turime 6 pavyzdines kasetes. Komandai reikia sukurti dar 10-15. Šie aprašymai yra pirmasis žingsnis.

---

## Prioritetų tvarka

Bandomajam laikotarpiui prioritetus dėliokite tokia tvarka:

1. **1 Užduotis** (VideoBlock) — 30 minučių, atblokiuoja video turinį
2. **2 Užduotis** (Validation CLI) — labai naudinga visiems tolesniems turinio darbams
3. **7 Užduotis** (Taksonomija) — duos pagrindą visam naujam turiniui
4. **9 Užduotis** (Scenarijų aprašymai) — suplanuoti prieš statant
5. **3, 4, 5 Užduotys** (Naujos užduotys) — pagrindinis turinio kūrimo konvejeris
6. **6 Užduotis** (Vaizdo ištekliai) — gali vykti lygiagrečiai su užduočių rašymu
7. **8 Užduotis** (Švari užduotis) — priklauso nuo V5 DI prompto, bet turinį galima parašyti jau dabar

---

*Šios užduotys sukurtos taip, kad būtų nepriklausomos viena nuo kitos. Kiekviena turi aiškų šabloną, kuriuo reikia sekti, pavyzdinį failą studijoms ir priėmimo kriterijus, pagal kuriuos galima pasitikrinti. Jei kažkas neaišku, klauskite užduoties (issue) komentaruose arba patikrinkite AUTHORING_GUIDE.md.*

---
