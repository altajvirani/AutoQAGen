from qa_generator import QAGenerator
from qa_generator import print_qa


class GenSubjQnsAns:
    def __init__(self, document, no_of_qns):
        self.document = document
        self.no_of_qns = no_of_qns

    def get_subj_qnsans(self):
        print('Please Wait....')
        qa_list = QAGenerator().generate(
            self.document, num_questions=self.no_of_qns, answer_style='sentences')
        subj_qns_ans = print_qa(qa_list)
        return subj_qns_ans

