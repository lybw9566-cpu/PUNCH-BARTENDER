import random  # <--- 引入随机库，解决推荐重复问题
import streamlit as st
import pandas as pd
import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 0. 门卫系统 ---
def check_access():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.title("🔒 这是一个私密应用")
    st.write("请在下方输入邀请码以继续访问。")
    
    user_input = st.text_input("请输入邀请码 (Access Key)", type="password")
    
    if st.button("解锁进入"):
        valid_keys = st.secrets.get("access_keys", [])
        if user_input in valid_keys:
            st.session_state.authenticated = True
            st.success("✅ 验证成功！正在加载...")
            st.rerun()
        else:
            st.error("❌ 邀请码无效")
    return False

if not check_access():
    st.stop()

# --- 1. 配置加载 ---
if "OPENAI_API_KEY" in st.secrets:
    API_KEY = st.secrets["OPENAI_API_KEY"]
    BASE_URL = st.secrets["OPENAI_BASE_URL"]
    MODEL_NAME = st.secrets["OPENAI_MODEL_NAME"]
else:
    load_dotenv()
    API_KEY = os.getenv("OPENAI_API_KEY")
    BASE_URL = os.getenv("OPENAI_BASE_URL")
    MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")

if not API_KEY:
    st.error("❌ 未配置 API Key")
    st.stop()

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 🔴 核心数据文件 (已翻译版)
DATA_FILE = "punch_recipes_cn.jsonl"

st.set_page_config(page_title="Punch AI 调酒师", page_icon="🍸", layout="wide") 

# --- 2. 数据加载与向量化 ---
@st.cache_resource
def load_data_and_vectors():
    data = []
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except FileNotFoundError:
        return None, None, None

    df = pd.DataFrame(data)

    # 混合文本用于搜索
    df['combined_text'] = (
        df['title'].fillna('') + " " + 
        df['ingredients'].astype(str) + " " + 
        df['tags'].astype(str)
    )

    # 模糊搜索配置
    vectorizer = TfidfVectorizer(
        stop_words='english',
        analyzer='char_wb', 
        ngram_range=(3, 5),
        max_features=5000
    )
    
    tfidf_matrix = vectorizer.fit_transform(df['combined_text'])

    return df, vectorizer, tfidf_matrix

df, vectorizer, tfidf_matrix = load_data_and_vectors()

if df is None:
    st.error(f"❌ 找不到数据文件 {DATA_FILE}")
    st.stop()

# --- 3. 核心 AI 逻辑 (含鱼塘扩容 + 随机洗牌) ---
def get_ai_recommendation(user_query):
    # === A. 检索 ===
    try:
        user_vec = vectorizer.transform([user_query])
        similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()
        
        # 🔴 扩大候选池到 100
        top_k = 100 
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # 🔴 随机洗牌：从前100名中随机抽20个
        candidates_pool = top_indices.tolist()
        
        if len(candidates_pool) > 20:
            selected_indices = random.sample(candidates_pool, 20)
        else:
            selected_indices = candidates_pool
            
        candidates = df.iloc[selected_indices]

    except Exception as e:
        return f"检索系统出错了: {e}", pd.DataFrame()

    # === B. 增强 ===
    context_text = ""
    for idx, row in candidates.iterrows():
        # 处理 instructions 可能是列表的情况
        inst_str = row['instructions']
        if isinstance(inst_str, list):
            inst_str = "\n".join(inst_str) # 变成字符串给AI看

        context_text += f"""
        [酒名: {row['title']}]
        [原料: {row['ingredients']}]
        [步骤: {inst_str}]
        [简介: {row.get('intro_philosophy', '')[:100]}]
        ---
        """

    # === C. 生成 ===
    combined_prompt = f"""
    【角色设定】
    你是一位见多识广的调酒师，擅长发掘冷门佳酿。
    
    【任务】
    用户想喝："{user_query}"
    从下面的【候选酒单】中，挑选 3 款推荐给用户。
    
    【策略要求】
    1. **不要总是推荐最常见的酒**。如果候选名单里有独特、冷门但符合用户口味的配方，优先推荐它们，给用户惊喜。
    2. 如果有多种基酒选择，请展示多样性。
    3. 基于提供的数据直接回答。
    
    【候选酒单】
    {context_text}

    【回复格式】
    ### 🍸 [酒名] (保持英文原名)
    - **推荐理由**: ...
    - **原料**: ...
    - **步骤**: ...
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME, 
            messages=[{"role": "user", "content": combined_prompt}],
            temperature=0.85, # 稍微再调高一点点，增加多样性
            max_tokens=4096, 
            presence_penalty=0.6 
        )
        if not response.choices:
            return f"⚠️ API 返回空结果。", candidates
        return response.choices[0].message.content, candidates

    except Exception as e:
        return f"❌ AI 连接报错: {str(e)}", pd.DataFrame()

# ==========================================
# 🎨 界面布局
# ==========================================

st.title("🍸 Punch AI 调酒师")

# --- 🔍 侧边栏 ---
with st.sidebar:
    st.header("📖 配方百科全书")
    search_query = st.text_input("🔍 搜索配方 (支持模糊拼写)", placeholder="例如: Bronx")
    
    selected_recipe_id = None
    
    if search_query:
        search_vec = vectorizer.transform([search_query])
        sims = cosine_similarity(search_vec, tfidf_matrix).flatten()
        top_indices = sims.argsort()[-10:][::-1]
        
        options_map = {}
        for i in top_indices:
            row = df.iloc[i]
            if sims[i] > 0.1:
                options_map[f"{row['title']}"] = i
        
        if options_map:
            st.success(f"找到 {len(options_map)} 个结果:")
            selected_name = st.selectbox("👇 点击选择查看详情:", list(options_map.keys()))
            if selected_name:
                selected_recipe_id = options_map[selected_name]
        else:
            st.warning("🤔 未找到相似配方")

# --- 📋 主界面：配方详情卡片 ---
if selected_recipe_id is not None:
    recipe_data = df.iloc[selected_recipe_id]
    
    with st.container(border=True):
        col_close, col_title = st.columns([1, 9])
        with col_close:
            if st.button("❌", key="close_btn"):
                selected_recipe_id = None
                st.rerun()

        with col_title:
            st.header(f"🍹 {recipe_data['title']}")
        
        st.info(f"💡 {recipe_data.get('intro_philosophy', '暂无简介')}")
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🧂 原料 Ingredients")
            ings = recipe_data['ingredients']
            # 判断原料是列表还是字符串
            if isinstance(ings, list):
                for ing in ings:
                    st.write(f"• {ing}")
            else:
                st.write(ings)
                    
        with c2:
            st.subheader("🥣 做法 Instructions")
            raw_inst = recipe_data['instructions']
            
            # 🔴 关键修复：智能判断步骤格式
            # 如果是列表，就遍历打印，去掉前面的索引 0: 1: ...
            if isinstance(raw_inst, list):
                for step in raw_inst:
                    st.write(step) 
            # 如果是字典(极少数情况)，取值打印
            elif isinstance(raw_inst, dict):
                for _, v in raw_inst.items():
                    st.write(v)
            # 如果是普通字符串，直接打印
            else:
                st.write(raw_inst)
            
        st.caption(f"标签: {recipe_data.get('tags', 'Classic')}")
        
    st.markdown("---") 

# --- 💬 聊天区域 ---
st.caption(f"私人定制 · {MODEL_NAME}")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "您好！我是您的 AI 侍酒师。您可以直接点餐，或者在左侧查阅配方。"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("今天想喝点什么风味的？"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("正在配方库中搜寻..."):
            ai_reply, related = get_ai_recommendation(prompt)
            st.markdown(ai_reply)
    st.session_state.messages.append({"role": "assistant", "content": ai_reply})