from fastapi import FastAPI
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import nltk, re
nltk.download("punkt")

app = FastAPI(title="Blog Writer")
model_id = "google/flan-t5-base"
tok = AutoTokenizer.from_pretrained(model_id)
mdl = AutoModelForSeq2SeqLM.from_pretrained(model_id)
gen = pipeline("text2text-generation", model=mdl, tokenizer=tok)

def clamp(x, lo, hi): return max(lo, min(hi, x))

@app.get("/outline")
def outline(topic: str, audience: str="tech", tone: str="informative", sections: int=5):
    sections = clamp(sections, 3, 10)
    prompt = (f"Create an SEO blog outline on '{topic}' for a {audience} audience, "
              f"tone '{tone}'. Include H2 sections with 1-2 H3 bullets each.")
    out = gen(prompt, max_length=350)[0]["generated_text"]
    return {"outline": out}

@app.get("/keywords")
def keywords(topic: str, n: int=10):
    n = clamp(n, 5, 20)
    prompt = f"List {n} SEO keywords and long-tail phrases for blog topic: {topic}. Return comma-separated."
    out = gen(prompt, max_length=120)[0]["generated_text"]
    kws = [k.strip() for k in re.split(r"[,\n]", out) if k.strip()]
    return {"keywords": kws[:n]}

@app.post("/draft")
def draft(topic: str, outline: str, tone: str="neutral", words: int=800):
    words = clamp(words, 300, 1800)
    prompt = (f"Write a {words}-word blog post on '{topic}' using this outline:\n{outline}\n"
              f"Tone: {tone}. Include intro, subsections with headings, and a conclusion.")
    out = gen(prompt, max_length=1024)[0]["generated_text"]
    return {"draft": out}
