from qa_generator import GapFillQAGenerator

class GenGapQnsAns:
    def __init__(self, document, no_of_qns):
        self.document = document
        self.no_of_qns = no_of_qns

    def get_gap_qnsans(self):
        print('Please Wait....')
        gap_qns_ans = GapFillQAGenerator(
            self.document, self.no_of_qns).generate_test()
        return gap_qns_ans
