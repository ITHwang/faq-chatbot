class PromptTemplate:
    def __init__(self, example_selector, memory, cannot_answer_comment):
        self.example_selector = example_selector
        self.memory = memory
        self.client_name = "고객"
        self.cannot_answer_comment = cannot_answer_comment

    def prompt(self, query, topk=10, n_history=3, thres=0.165, min_query_len=20):
        # validate
        examples = self.example_selector.select(query, topk=topk, thres=thres)
        if not examples:
            return self.cannot_answer_comment

        if len(query) < min_query_len:
            n_repeat = (min_query_len // len(query)) + 1
            query = " ".join([query for _ in range(n_repeat)])

        # query again
        example_query = (
            f"previous question answering:\n{str(self.memory[-1])} so {query}"
            if len(self.memory) >= 1
            else query
        )
        examples = self.example_selector.select(example_query, topk=topk, thres=thres)

        histories = self.memory.get(n_history)

        example_text = "\n".join([str(e) for e in examples])
        history_text = (
            "\n".join([str(h) for h in histories])
            if histories
            else "사실 이전 상담 내용이 없어. 고객의 첫 질문이야.\n"
        )

        prompt_text = f"""
        너는 네이버 스마트스토어 고객센터 상담원이야.
        {self.client_name}님이 상담을 요청했어.
        너는 {self.client_name}님에게 친절하고 상냥하게 답변해야 해.
        이전 상담 내용은 아래와 같아.

        {history_text}

        {n_history}가지 답변 예시를 아래에 보여줄게.

        {example_text}

        자 이제 사용자의 질문을 아래에 줄테니, 이전 상담 내용과 답변 예시에 기반해서 답변해줘.

        질문: {query}
        대답:
        """

        return prompt_text
