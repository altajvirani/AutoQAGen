
from flask import Flask, request
from flask_cors import CORS
from gen_bool_qnsans import GenBoolQnsAns
from gen_gap_qnsans import GenGapQnsAns
from gen_mcq_qnsans import GenMcqQnsAns
from gen_subj_qnsans import GenSubjQnsAns
from gen_wh_qnsans import GenWhQnsAns


def return_bool_qnsans(document, no_of_qns):
    bool_qnsans = GenBoolQnsAns(document, no_of_qns).get_bool_qnsans()
    return bool_qnsans


def return_gap_qnsans(document, no_of_qns):
    gap_qnsans = GenGapQnsAns(document, no_of_qns).get_gap_qnsans()
    return gap_qnsans


def return_mcq_qnsans(document, no_of_qns):
    mcq_qnsans = GenMcqQnsAns(document, no_of_qns).get_mcq_qnsans()
    return mcq_qnsans


def return_subj_qnsans(document, no_of_qns):
    subj_qnsans = GenSubjQnsAns(document, no_of_qns).get_subj_qnsans()
    return subj_qnsans


def return_wh_qnsans(document, no_of_qns):
    wh_qnsans = GenWhQnsAns(document, no_of_qns).get_wh_qnsans()
    return wh_qnsans


app = Flask(__name__)
CORS(app)


@app.route('/send_doc', methods=['POST'])
def received_doc():
    data = request.get_json()
    document = str(data['document'])
    no_of_qns = int(data['no_of_qns'])
    res = {}

    if ('single_answer' in data['qn_type']):
        res['single_answer'] = return_wh_qnsans(document, no_of_qns)
    if ('gap_fill' in data['qn_type']):
        res['gap_fill'] = return_gap_qnsans(document, no_of_qns)
    if ('mcq' in data['qn_type']):
        res['mcq'] = return_mcq_qnsans(document, no_of_qns)
    if ('boolean' in data['qn_type']):
        res['boolean'] = return_bool_qnsans(document, no_of_qns)
    if ('subjective' in data['qn_type']):
        res['subjective'] = return_subj_qnsans(document, no_of_qns)

    return res


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
