class QA:
    def __init__(self, question: str, answer: str):
        self.question = question
        self.answer = answer

    def __str__(self):
        return f"질문: {self.question}\n대답: {self.answer}\n"

    def __repr__(self):
        return self.__str__()
