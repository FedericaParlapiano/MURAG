PROMPT_TEMPLATES = {
    "en": """You are an intelligent reasoning agent. Answer the question by thinking step-by-step.
        
IMPORTANT RULE: If the input question is simple and direct (single-hop), do not overcomplicate it. Your first `Thought` should state that it's a direct question, and your first `Search` should simply be the original question itself (or its main keywords).

You must use the exact following format:
        
Question: the input question you must answer
Thought: your internal reasoning about what you need to find out next
Search: a short, specific search query to look up information (write this in English)
Observation: the result of the search
        
... (this Thought/Search/Observation process can repeat up to {max_hops} times) ...
        
Thought: your final reasoning step where you synthesize all the observations to deduce the exact answer
Final Answer: ONLY the short, exact answer (e.g., just the specific name, date, or entity). Do not write a full sentence. Write the final answer in {response_language}.
        
---
Question: {question}""",

    "ar": """أنت وكيل ذكي للتفكير المنطقي. أجب عن السؤال بالتفكير خطوة بخطوة.

قاعدة مهمة: إذا كان السؤال بسيطًا ومباشرًا (خطوة واحدة)، فلا تعقده. يجب أن يوضح `Thought` الأول أنه سؤال مباشر، ويجب أن يكون `Search` الأول ببساطة هو السؤال الأصلي نفسه (أو كلماته الرئيسية).

يجب عليك استخدام التنسيق الدقيق التالي:

Question: السؤال الذي يجب عليك الإجابة عليه
Thought: تفكيرك الداخلي حول ما تحتاج إلى معرفته بعد ذلك
Search: استعلام بحث قصير ومحدد للبحث عن المعلومات (اكتب هذا باللغة الإنجليزية)
Observation: نتيجة البحث

... (يمكن أن تتكرر عملية Thought/Search/Observation هذه حتى {max_hops} مرات) ...

Thought: خطوة تفكيرك النهائية حيث تقوم بتجميع كل الملاحظات لاستنتاج الإجابة الدقيقة
Final Answer: فقط الإجابة القصيرة والدقيقة (مثل الاسم المحدد، التاريخ، أو الكيان). لا تكتب جملة كاملة. اكتب الإجابة النهائية باللغة {response_language}.

---
Question: {question}""",

    "ru": """Вы — интеллектуальный агент. Ответьте на вопрос, рассуждая шаг за шагом.

ВАЖНОЕ ПРАВИЛО: Если входной вопрос простой и прямой (один шаг), не усложняйте его. Ваш первый `Thought` должен констатировать, что это прямой вопрос, а ваш первый `Search` должен быть просто самим исходным вопросом (или его ключевыми словами).

Вы должны использовать строго следующий формат:

Question: входной вопрос, на который вы должны ответить
Thought: ваши внутренние рассуждения о том, что вам нужно узнать дальше
Search: короткий, конкретный поисковый запрос для поиска информации (напишите его на английском языке)
Observation: результат поиска

... (этот процесс Thought/Search/Observation может повторяться до {max_hops} раз) ...

Thought: ваш финальный шаг рассуждений, где вы синтезируете все наблюдения, чтобы вывести точный ответ
Final Answer: ТОЛЬКО короткий, точный ответ (например, только конкретное имя, дата или сущность). Не пишите полное предложение. Напишите окончательный ответ на языке {response_language}.

---
Question: {question}""",

    "zh": """你是一个智能推理代理。请逐步思考并回答问题。

重要规则：如果输入的问题是简单直接的（单步），请不要把它复杂化。你的第一个 `Thought` 应该说明这是一个直接的问题，而你的第一个 `Search` 应该简单地使用原始问题本身（或其主要关键词）。

你必须使用以下完全一致的格式：

Question: 你必须回答的输入问题
Thought: 你关于接下来需要找出什么信息的内部推理
Search: 用于查找信息的简短、具体的搜索查询（请务必用英语编写）
Observation: 搜索结果

... (这个 Thought/Search/Observation 过程最多可以重复 {max_hops} 次) ...

Thought: 你的最终推理步骤，你需要综合所有观察结果来推断出确切的答案
Final Answer: 仅提供简短、确切的答案（例如，仅提供特定的名称、日期或实体）。不要写完整的句子。请用 {response_language} 写出最终答案。

---
Question: {question}""",

    "es": """Eres un agente de razonamiento inteligente. Responde a la pregunta pensando paso a paso.
        
REGLA IMPORTANTE: Si la pregunta de entrada es simple y directa (de un solo salto), no la compliques en exceso. Tu primer `Thought` debe indicar que es una pregunta directa, y tu primer `Search` debe ser simplemente la pregunta original en sí (o sus palabras clave principales).

Debes usar exactamente el siguiente formato:
        
Question: la pregunta de entrada que debes responder
Thought: tu razonamiento interno sobre lo que necesitas averiguar a continuación
Search: una consulta de búsqueda corta y específica para buscar información (escribe esto en inglés)
Observation: el resultado de la búsqueda
        
... (este proceso de Thought/Search/Observation puede repetirse hasta {max_hops} veces) ...
        
Thought: tu paso de razonamiento final donde sintetizas todas las observaciones para deducir la respuesta exacta
Final Answer: SOLAMENTE la respuesta corta y exacta (por ejemplo, solo el nombre, fecha o entidad específica). No escribas una oración completa. Escribe la respuesta final en {response_language}.
        
---
Question: {question}""",

    "de": """Du bist ein intelligenter Agent für logisches Denken. Beantworte die Frage, indem du Schritt für Schritt überlegst.
        
WICHTIGE REGEL: Wenn die eingegebene Frage einfach und direkt ist (Single-Hop), mache sie nicht unnötig kompliziert. Dein erster `Thought` sollte angeben, dass es sich um eine direkte Frage handelt, und dein erster `Search` sollte einfach die ursprüngliche Frage selbst (oder ihre Hauptschlagwörter) sein.

Du musst exakt das folgende Format verwenden:
        
Question: die eingegebene Frage, die du beantworten musst
Thought: deine internen Überlegungen darüber, was du als Nächstes herausfinden musst
Search: eine kurze, spezifische Suchanfrage, um Informationen nachzuschlagen (schreibe dies auf Englisch)
Observation: das Ergebnis der Suche
        
... (dieser Prozess aus Thought/Search/Observation kann bis zu {max_hops}-mal wiederholt werden) ...
        
Thought: dein finaler Denkschritt, in dem du alle Beobachtungen zusammenfasst, um die genaue Antwort abzuleiten
Final Answer: NUR die kurze, exakte Antwort (z. B. nur der spezifische Name, das Datum oder die Entität). Schreibe keinen vollständigen Satz. Schreibe die endgültige Antwort auf {response_language}.
        
---
Question: {question}"""
}

def genera_prompt(lang_code, question, max_hops, response_language_name):
    template = PROMPT_TEMPLATES.get(lang_code, PROMPT_TEMPLATES["en"])
    return template.format(
        question=question,
        max_hops=max_hops,
        response_language=response_language_name
    )
