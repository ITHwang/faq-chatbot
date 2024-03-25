from transformers import LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM

class QuestionAnsweringModel():
    def __init__(self, model_name):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def answer(self, question, max_length=100, device="gpu"):
        input_text = self.prompt(question)
        inputs = self.tokenizer(input_text, return_tensors='pt').to(device)
        outputs = self.model.to(device).generate(inputs.input_ids, max_length=max_length).to("cpu")
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
        # todo: print answer char by char
    
    def prompt(self, question):
        return f"질문: {question}\n답변: "