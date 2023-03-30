from qa_generator import QAGenerator
from qa_generator import print_qa


class GenWhQnsAns:
    def __init__(self, document, no_of_qns):
        self.document = document
        self.no_of_qns = no_of_qns

    def get_wh_qnsans(self):
        print("Please Wait....")
        qa_list = QAGenerator().generate(
            self.document, num_questions=self.no_of_qns, answer_style="multiple_choice")
        wh_qns_ans = print_qa(qa_list, isSingle=True)
        return wh_qns_ans
