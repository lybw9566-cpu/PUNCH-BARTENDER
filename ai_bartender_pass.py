import random  # <--- 别忘了在文件最顶部的 import 区域加上这句
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
DATA_FILE = "punch_recipes.jsonl"

st.set_page_config(page_title="Punch AI 调酒师", page_icon="🍸", layout="wide") 
# 注意：layout 改为 'wide' 可以让侧边栏和主内容更宽敞

# --- 2. 数据加载与向量化 (保持 char_wb 模糊搜索) ---
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

    # 使用字符级 n-gram 实现模糊匹配
    vectorizer = TfidfVectorizer(
        stop_words='english',
        analyzer='char_wb', 
        ngram_range=(3, 5)
    )
    
    tfidf_matrix = vectorizer.fit_transform(df['combined_text'])

    return df, vectorizer, tfidf_matrix

df, vectorizer, tfidf_matrix = load_data_and_vectors()

if df is None:
    st.error(f"❌ 找不到数据文件 {DATA_FILE}")
    st.stop()

# --- 3. 核心 AI 逻辑 (升级版：增加随机多样性) ---
def get_ai_recommendation(user_query):
    # === A. 检索 ===
    try:
        user_vec = vectorizer.transform([user_query])
        similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()
        
        # 🔴 关键修改 1: 扩大候选池 (鱼塘)
        # 以前我们只取前 30 (argsort()[-30:])，它们永远是固定的。
        # 现在我们取前 100 个，这些都是相关性不错的酒。
        top_k = 100 
        
        # 获取前 100 名的索引 (从低到高，所以后面要切片)
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # 🔴 关键修改 2: 随机洗牌 (Shuffling)
        # 将这 top_indices 转为列表
        candidates_pool = top_indices.tolist()
        
        # 从这 100 个里，随机抽取 20 个给 AI
        # 这样既保证了相关性(都在前100)，又保证了每次不一样
        if len(candidates_pool) > 20:
            selected_indices = random.sample(candidates_pool, 20)
        else:
            selected_indices = candidates_pool
            
        candidates = df.iloc[selected_indices]

    except Exception as e:
        return f"检索系统出错了: {e}", pd.DataFrame()

    # === B. 增强 (构建 Context) ===
    context_text = ""
    for idx, row in candidates.iterrows():
        # 这里适配了中文数据库的字段，如果是英文版会自动显示英文
        context_text += f"""
        [酒名: {row['title']}]
        [原料: {row['ingredients']}]
        [步骤: {row['instructions']}]
        [简介: {row.get('intro_philosophy', '')[:100]}]
        ---
        """

    # === C. 生成 (Prompt) ===
    combined_prompt = f"""
    【角色设定】
    你是一位见多识广的调酒师，擅长发掘冷门佳酿。
    
    【任务】
    用户想喝："{user_query}"
    从下面的【候选酒单】中，挑选 3 款推荐给用户。
    
    【策略要求】
    1. **不要总是推荐最常见的酒**。如果候选名单里有独特、冷门但符合用户口味的配方，优先推荐它们，给用户惊喜。
    2. 如果有多种基酒选择（如既有金酒又有伏特加），请展示多样性。
    
    【候选酒单】
    {context_text}

    【回复格式】
    请用优雅的中文回复。
    ### 🍸 [酒名]
    - **推荐理由**: ...
    - **原料**: ...
    - **步骤**: ...
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME, 
            messages=[{"role": "user", "content": combined_prompt}],
            temperature=0.8, # 稍微调高温度，让 AI 说话更有创造力
            max_tokens=4096, 
            presence_penalty=0.6 # 惩罚重复内容
        )
        if not response.choices:
            return f"⚠️ API 返回空结果。", candidates
        return response.choices[0].message.content, candidates

    except Exception as e:
        return f"❌ AI 连接报错: {str(e)}", pd.DataFrame()
# ==========================================
# 🎨 界面布局开始
# ==========================================

st.title("🍸 Punch AI 调酒师")

# --- 🔍 侧边栏：超级模糊搜索 ---
with st.sidebar:
    st.header("📖 配方百科全书")
    # 1. 搜索框
    search_query = st.text_input("🔍 搜索配方 (支持模糊拼写)", placeholder="例如: Bronx 或 margrita")
    
    selected_recipe_id = None
    
    if search_query:
        # 复用那个强大的向量搜索引擎
        # 即使你输错 "Mrgarita"，它也能算出它是 Margarita
        search_vec = vectorizer.transform([search_query])
        sims = cosine_similarity(search_vec, tfidf_matrix).flatten()
        
        # 找出最相似的 10 个
        top_indices = sims.argsort()[-10:][::-1]
        
        # 制作下拉菜单选项字典: { "酒名": ID }
        options_map = {}
        for i in top_indices:
            row = df.iloc[i]
            # 如果相似度太低(小于0.1)，可能是噪音，不显示
            if sims[i] > 0.1:
                options_map[f"{row['title']}"] = i
        
        if options_map:
            st.success(f"找到 {len(options_map)} 个相关结果:")
            # 2. 下拉选择框
            selected_name = st.selectbox("👇 点击选择查看详情:", list(options_map.keys()))
            
            if selected_name:
                selected_recipe_id = options_map[selected_name]
        else:
            st.warning("🤔 未找到相似配方，请换个词试试")

# ==========================================
# 📋 主界面：智能翻译配方卡片
# ==========================================
if selected_recipe_id is not None:
    # 1. 获取原始英文数据
    raw_data = df.iloc[selected_recipe_id]
    
    # 2. 构建翻译请求 Prompt
    translation_prompt = f"""
    【任务】
    请将以下鸡尾酒配方翻译成中文，并按照 Markdown 格式排版。
    
    【原始数据】
    Name: {raw_data['title']}
    Intro: {raw_data['intro_philosophy']}
    Ingredients: {raw_data['ingredients']}
    Instructions: {raw_data['instructions']}
    Tags: {raw_data.get('tags', '')}

    【要求】
    1. 标题用 H2 (##) 加 emoji。
    2. 简介用引用格式 (>)。
    3. 原料用列表，保留原始用量（如 2 oz），但在括号里估算 ml 数（1 oz ≈ 30ml）。
    4. 步骤必须清晰易懂。
    5. 语气：像一位优雅的调酒师在介绍。
    """

    # 3. 显示加载动画并调用 AI
    with st.container(border=True):
        # 如果用户频繁点击，每次都翻译有点浪费，但在 Streamlit 里这是最简单的写法
        # 如果你介意速度，可以使用 @st.cache_data 缓存翻译结果
        
        with st.spinner(f"正在将 {raw_data['title']} 翻译为中文..."):
            try:
                trans_response = client.chat.completions.create(
                    model=MODEL_NAME, # 使用 gpt-4o-mini 速度极快
                    messages=[{"role": "user", "content": translation_prompt}],
                    temperature=0.3, # 翻译需要准确，温度调低
                    max_tokens=2000
                )
                translated_content = trans_response.choices[0].message.content
                
                # 4. 展示翻译后的结果
                # 关闭按钮 (其实只是清空选中状态，但在 Streamlit 需要重新加载)
                col1, col2 = st.columns([9, 1])
                with col2:
                    if st.button("❌", help("关闭卡片")):
                        selected_recipe_id = None
                        st.rerun()
                
                # 渲染 AI 写好的 Markdown
                st.markdown(translated_content)
                
                # 5. 在底部显示原始英文（折叠），方便核对
                with st.expander("🔍 查看原始英文配方 (Original Recipe)"):
                    st.write(raw_data.to_dict())
                    
            except Exception as e:
                st.error(f"翻译服务开小差了: {e}")
                # 如果翻译失败，兜底显示英文
                st.write(raw_data)

    st.markdown("---") # 分割线

# --- 💬 聊天区域 (AI 调酒师) ---
st.caption(f"私人定制 · {MODEL_NAME}")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "您好！您可以在左侧搜索特定的配方卡片，也可以直接在这里告诉我您的口味，让我为您推荐。"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("描述您的口味，或让 AI 推荐..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("AI 正在思考..."):
            ai_reply, related = get_ai_recommendation(prompt)
            st.markdown(ai_reply)
    st.session_state.messages.append({"role": "assistant", "content": ai_reply})