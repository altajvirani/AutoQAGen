from qa_generator import BoolQAGenerator

class GenBoolQnsAns:
    def __init__(self, document, no_of_qns):
        self.document = document
        self.no_of_qns = no_of_qns 

    def get_bool_qnsans(self):
        bool_qns_ans = BoolQAGenerator().get_bool_qnsans(self.document, self.no_of_qns)
        return bool_qns_ans