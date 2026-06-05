# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, redirect, url_for, jsonify
from datetime import datetime, timedelta, date
import json

app = Flask(__name__)

# ============================================================
# 学科
# ============================================================
SUBJECTS = {
    "english":    {"name": "英语",   "color": "#5b9cf5", "icon": "fas fa-flag-usa"},
    "psychology": {"name": "心理学", "color": "#e85d75", "icon": "fas fa-brain"},
    "cognitive":  {"name": "认知学", "color": "#b388ff", "icon": "fas fa-lightbulb"},
    "business":   {"name": "商业",   "color": "#f5a623", "icon": "fas fa-chart-line"},
}

# ============================================================
# 打卡系统配置
# ============================================================
CHECKPOINTS = [
    {"id":"review",   "name":"间隔复习",   "xp":3, "icon":"fas fa-clock"},
    {"id":"duolingo", "name":"多邻国学习", "xp":3, "icon":"fas fa-language"},
    {"id":"reading",  "name":"文章研读",   "xp":3, "icon":"fas fa-book-open"},
    {"id":"practice", "name":"针对性练习", "xp":1, "icon":"fas fa-pen-to-square"},
]
ALL_FOUR_BONUS = 2
DAILY_MAX_XP = 12

VOCAB_SOURCES = ["duolingo", "reading"]  # which checkpoints collect vocab

STREAK_MILESTONES = {7:10, 14:15, 30:30, 60:50, 100:100, 365:365}

CHARACTER_MILESTONES = {
    5:   ("列兵",   "fas fa-chess-pawn"),
    10:  ("骑士",   "fas fa-chess-knight"),
    15:  ("将军",   "fas fa-chess-queen"),
    20:  ("统帅",   "fas fa-chess-king"),
    30:  ("精英",   "fas fa-star"),
    45:  ("大师",   "fas fa-crown"),
    60:  ("传奇",   "fas fa-gem"),
    90:  ("史诗",   "fas fa-dragon"),
    120: ("神话",   "fas fa-fire"),
    150: ("圣者",   "fas fa-book-bible"),
    180: ("帝皇",   "fas fa-chess-rook"),
    210: ("宗师",   "fas fa-wand-magic-sparkles"),
    240: ("觉醒",   "fas fa-eye"),
    270: ("创世",   "fas fa-cloud-moon"),
    300: ("升华",   "fas fa-feather-pointed"),
    330: ("永恒",   "fas fa-infinity"),
    360: ("圆满",   "fas fa-sun"),
}

# ============================================================
# 词汇 & 单元存储
# ============================================================
# vocab_items: [{id, subject, word, meaning, source, unit_id, date, mastery(0-3),
#                times_tested(0), times_correct(0), last_tested(None)}]
vocab_items = []
_vocab_counter = 0
def next_vid():
    global _vocab_counter
    _vocab_counter += 1
    return _vocab_counter

# vocab_units: [{id, subject, title, date, type(duolingo/reading), word_count}]
vocab_units = []
_vunit_counter = 0
def next_uid():
    global _vunit_counter
    _vunit_counter += 1
    return _vunit_counter

# ============================================================
# 用户状态
# ============================================================
class SubjectState:
    def __init__(self):
        self.total_xp = 0; self.streak = 0; self.freeze_cards = 3
        self.last_checkin = None; self.weekly_xp = 0; self.weekly_target = 70
        self.monthly_xp = 0; self.week_start = None; self.used_freeze_this_week = False

state = {key: SubjectState() for key in SUBJECTS}
check_ins = []

# ============================================================
# 工具
# ============================================================
def today_str():
    return datetime.now().strftime("%Y-%m-%d")

def get_today_checkin(subject_key):
    ts = today_str()
    for c in check_ins:
        if c["date"] == ts and c["subject"] == subject_key:
            return c
    return None

def get_week_start():
    t = date.today()
    return (t - timedelta(days=t.weekday())).strftime("%Y-%m-%d")

def get_streak_fire_class(st):
    if st.last_checkin != today_str(): return "gray"
    elif st.streak >= 10: return "crimson"
    else: return "red"

def build_day_nodes(subject_key):
    st = state[subject_key]
    subject_days = len({c["date"] for c in check_ins if c["subject"] == subject_key})
    next_day = subject_days + 1

    nodes = []
    for day in range(1, 361):
        is_milestone = day in CHARACTER_MILESTONES
        mile_name, mile_icon = CHARACTER_MILESTONES.get(day, ("", ""))
        if day <= subject_days: status = "done"
        elif day == next_day and day <= 360: status = "unlocked"
        else: status = "locked"
        nodes.append({"day":day, "status":status, "is_milestone":is_milestone,
                       "mile_name":mile_name, "mile_icon":mile_icon})
    return nodes

def build_chart_data(subject_key, days=30):
    today = date.today()
    xp_by_date = {}
    for c in check_ins:
        if c["subject"] == subject_key:
            xp_by_date[c["date"]] = xp_by_date.get(c["date"], 0) + c["xp"]
    return [{"date":(today - timedelta(days=i)).strftime("%Y-%m-%d"),
             "xp":xp_by_date.get((today - timedelta(days=i)).strftime("%Y-%m-%d"), 0)}
            for i in range(days-1, -1, -1)]

def get_vocab_stats(subject_key):
    """Return mastery distribution and source counts for a subject"""
    items = [v for v in vocab_items if v["subject"] == subject_key]
    total = len(items)
    mastery_dist = {0:0,1:0,2:0,3:0}
    source_count = {"duolingo":0, "reading":0}
    for v in items:
        mastery_dist[v["mastery"]] = mastery_dist.get(v["mastery"], 0) + 1
        source_count[v["source"]] = source_count.get(v["source"], 0) + 1
    return total, mastery_dist, source_count

def get_vocab_units_data(subject_key):
    return [u for u in vocab_units if u["subject"] == subject_key]

# ============================================================
# 模拟数据
# ============================================================
def mock_tasks():
    return [
        {"id":1, "title":"间隔复习到期词汇","subject":"english","target":20,"current":20,"status":"done","duration":"10分钟","type":"daily"},
        {"id":2, "title":"多邻国完成今日单元","subject":"english","target":1,"current":1,"status":"done","duration":"15分钟","type":"daily"},
        {"id":3, "title":"精读英文文章并记录生词","subject":"english","target":1,"current":0,"status":"pending","duration":"15分钟","type":"daily"},
        {"id":4, "title":"完成今日针对性练习","subject":"english","target":1,"current":0,"status":"pending","duration":"5分钟","type":"daily"},
    ]

def mock_achievements():
    return [
        {"id":1,"name":"连续7天","desc":"连续学习7天","score":50,"level":"copper","cat":"streak","earned":True},
        {"id":2,"name":"学习达人","desc":"完成10个关卡","score":100,"level":"silver","cat":"study","earned":True},
        {"id":3,"name":"连续30天","desc":"连续打卡30天","score":200,"level":"gold","cat":"streak","earned":True},
        {"id":4,"name":"满分达成","desc":"单次评分5分","score":300,"level":"gold","cat":"skill","earned":False},
        {"id":5,"name":"多面手","desc":"完成所有学科","score":400,"level":"diamond","cat":"study","earned":False},
        {"id":6,"name":"早起鸟","desc":"连续7天早上学习","score":150,"level":"silver","cat":"streak","earned":True},
        {"id":7,"name":"深度思考","desc":"写5篇学习笔记","score":80,"level":"copper","cat":"skill","earned":False},
        {"id":8,"name":"多邻国魂","desc":"同时学4门学科","score":500,"level":"legend","cat":"study","earned":False},
    ]

ACHIEVEMENT_LEVELS = {
    "copper":{"name":"铜牌","color":"#CD7F32","icon":"🥉"},
    "silver":{"name":"银牌","color":"#C0C0C0","icon":"🥈"},
    "gold":{"name":"金牌","color":"#FFD700","icon":"🥇"},
    "diamond":{"name":"钻石","color":"#B9F2FF","icon":"💎"},
    "legend":{"name":"传说","color":"#9400D3","icon":"👑"},
}

def mock_persona():
    return {
        "radar":[{"label":"词汇","score":0.65},{"label":"听力","score":0.85},{"label":"口语","score":0.72},{"label":"阅读","score":0.78},{"label":"写作","score":0.50}],
        "style":{"visual":65,"auditory":20,"kinesthetic":15},
        "time_dist":[{"label":"6-9点","pct":15},{"label":"9-12点","pct":35},{"label":"14-18点","pct":25},{"label":"19-22点","pct":20},{"label":"22-24点","pct":5}],
        "insights":["您的词汇量是薄弱环节，建议加强单词背诵","您的最佳学习时间是上午9-12点","您的学习效率在连续学习30分钟后明显下降"],
    }

def mock_suggestions():
    return [
        {"id":1,"priority":"P0","title":"词汇量增长停滞","heat":3,"current":"每天20个单词","suggest":"每天30个单词","effect":"2周内词汇量提升15%"},
        {"id":2,"priority":"P1","title":"听力练习时间不当","heat":2,"current":"晚上8点练习听力","suggest":"调整到上午9-11点","effect":"听力准确率提升10%"},
        {"id":3,"priority":"P2","title":"缺少输出训练","heat":1,"current":"只进行输入练习","suggest":"增加口语和写作输出","effect":"综合能力更均衡"},
    ]

# ============================================================
# 路由
# ============================================================
@app.route("/")
def home():
    ws = get_week_start()
    for key, st in state.items():
        if st.week_start != ws:
            st.week_start = ws; st.weekly_xp = 0; st.used_freeze_this_week = False

    day_nodes = {key: build_day_nodes(key) for key in SUBJECTS}
    today_ci = {key: get_today_checkin(key) for key in SUBJECTS}
    streak_fire = {key: get_streak_fire_class(state[key]) for key in SUBJECTS}
    chart_data = {key: build_chart_data(key) for key in SUBJECTS}

    vocab_stats = {}
    vocab_unit_list = {}
    for key in SUBJECTS:
        total, mastery_dist, source_count = get_vocab_stats(key)
        vocab_stats[key] = {"total":total, "mastery":mastery_dist, "sources":source_count}
        vocab_unit_list[key] = get_vocab_units_data(key)

    return render_template("index.html",
        subjects=SUBJECTS, day_nodes=day_nodes, chart_data=chart_data,
        state=state, today_ci=today_ci, streak_fire=streak_fire,
        checkpoints=CHECKPOINTS, streak_milestones=STREAK_MILESTONES,
        vocab_stats=vocab_stats, vocab_units=vocab_unit_list,
        tasks=mock_tasks(), achievements=mock_achievements(),
        ach_levels=ACHIEVEMENT_LEVELS, persona=mock_persona(),
        suggestions=mock_suggestions(),
        now=datetime.now().strftime("%Y-%m-%d %H:%M"), today=today_str(),
    )

@app.route("/checkin", methods=["POST"])
def do_checkin():
    key = request.form.get("subject_key", "english")
    if key not in SUBJECTS: return redirect(url_for("home"))

    ts = today_str()
    existing = get_today_checkin(key)
    if existing: return redirect(url_for("home"))

    st = state[key]
    xp, ck, full = 0, {}, True
    for cp in CHECKPOINTS:
        cid = cp["id"]
        done = request.form.get("chk_"+cid) == "1"
        ck[cid] = done
        if done: xp += cp["xp"]
        else: full = False
    if full: xp += ALL_FOUR_BONUS

    # Process vocabulary input for duolingo & reading
    for src in VOCAB_SOURCES:
        if ck.get(src):
            raw = request.form.get("vocab_"+src, "").strip()
            unit_title = request.form.get("unit_"+src, "").strip()
            if raw:
                lines = [l.strip() for l in raw.split("\n") if l.strip()]
                words = []
                for line in lines:
                    if " - " in line:
                        w, m = line.split(" - ", 1)
                        words.append({"word":w.strip(), "meaning":m.strip()})
                    elif "—" in line:
                        w, m = line.split("—", 1)
                        words.append({"word":w.strip(), "meaning":m.strip()})
                    else:
                        # single word, no meaning
                        words.append({"word":line.strip(), "meaning":""})

                if words:
                    uid = next_uid()
                    vocab_units.append({
                        "id":uid, "subject":key, "title":unit_title or f"{ts} {src}",
                        "date":ts, "type":src, "word_count":len(words)
                    })
                    for w in words:
                        vocab_items.append({
                            "id":next_vid(), "subject":key, "word":w["word"],
                            "meaning":w["meaning"], "source":src, "unit_id":uid,
                            "date":ts, "mastery":0,
                            "times_tested":0, "times_correct":0, "last_tested":None
                        })

    # Streak
    streak_before = st.streak
    use_freeze = request.form.get("use_freeze") == "1"
    yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    if st.last_checkin == yesterday: st.streak += 1
    elif st.last_checkin == ts: pass
    elif use_freeze and st.freeze_cards > 0:
        st.freeze_cards -= 1; st.used_freeze_this_week = True; st.streak += 1
    else: st.streak = 1

    st.last_checkin = ts
    st.total_xp += xp; st.weekly_xp += xp; st.monthly_xp += xp
    milestone_bonus = 0
    if st.streak in STREAK_MILESTONES:
        milestone_bonus = STREAK_MILESTONES[st.streak]
        st.total_xp += milestone_bonus; st.weekly_xp += milestone_bonus
    if not st.used_freeze_this_week and st.freeze_cards < 3: st.freeze_cards += 1

    check_ins.append({"id":len(check_ins)+1, "date":ts, "subject":key,
        "checkpoints":ck, "xp":xp, "streak_before":streak_before,
        "streak_after":st.streak, "used_freeze":use_freeze,
        "milestone_bonus":milestone_bonus})
    return redirect(url_for("home"))

# ─── Test API ───
@app.route("/api/test/start/<key>")
def api_test_start(key):
    """Generate a test set from all vocab for a subject, prioritizing weak words"""
    if key not in SUBJECTS: return jsonify({"error":"invalid subject"}), 404
    items = [v for v in vocab_items if v["subject"] == key]
    if not items: return jsonify({"error":"no vocabulary yet"}), 404

    mode = request.args.get("mode", "zh2en")  # zh2en, en2zh
    unit_id = request.args.get("unit_id")
    count = int(request.args.get("count", 10))

    pool = items
    if unit_id:
        pool = [v for v in items if v["unit_id"] == int(unit_id)]
        if not pool: return jsonify({"error":"unit has no words"}), 404

    # Sort by mastery asc (weakest first), then by times_tested asc
    pool.sort(key=lambda v: (v["mastery"], v["times_tested"]))

    # Take up to count, but at least include some mastered words for variety
    weak = [v for v in pool if v["mastery"] <= 1]
    strong = [v for v in pool if v["mastery"] >= 2]
    # Take up to count
    result = []
    # Mix: weak words first, but sprinkle some strong ones
    weak = [v for v in pool if v["mastery"] <= 1]
    ok = [v for v in pool if v["mastery"] == 2]
    strong = [v for v in pool if v["mastery"] == 3]
    result = weak[:int(count*0.6)] + ok[:int(count*0.25)] + strong[:int(count*0.15)]
    remaining = [v for v in pool if v not in result]
    result += remaining[:count - len(result)]
    result = result[:count]
    selected = result

    questions = []
    for v in selected:
        if mode == "zh2en":
            questions.append({"id":v["id"], "word":v["word"], "meaning":v["meaning"],
                              "prompt":v["meaning"], "answer":v["word"].lower(),
                              "current_mastery":v["mastery"]})
        else:  # en2zh
            questions.append({"id":v["id"], "word":v["word"], "meaning":v["meaning"],
                              "prompt":v["word"], "answer":v["meaning"].lower(),
                              "current_mastery":v["mastery"]})

    return jsonify({"questions":questions, "total":len(questions)})

@app.route("/api/test/submit", methods=["POST"])
def api_test_submit():
    """Submit test results. Body: {results: [{id, correct:bool}, ...]}"""
    data = request.get_json(force=True)
    results = data.get("results", [])
    for r in results:
        vid = r["id"]
        correct = r.get("correct", False)
        for v in vocab_items:
            if v["id"] == vid:
                v["times_tested"] += 1
                if correct:
                    v["times_correct"] += 1
                    if v["mastery"] < 3 and v["times_correct"] >= 3:
                        v["mastery"] += 1
                else:
                    v["mastery"] = max(0, v["mastery"] - 1)
                v["last_tested"] = today_str()
                break
    return jsonify({"ok":True})

# ============================================================
if __name__ == "__main__":
    print("\n>>> Smart Learning System <<<")
    print("    Open http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)
