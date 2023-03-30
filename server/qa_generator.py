import re
import nltk
import numpy as np
from nltk.corpus import wordnet as wn
import en_core_web_sm
import json
import random
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    T5ForConditionalGeneration,
    T5Tokenizer
)
from typing import (
    Any, 
    List, 
    Mapping, 
    Tuple)
from boolean_question import BoolQ
bq = BoolQ()

class QAGenerator:
    """A transformer-based NLP system for generating reading comprehension-style questions from
    texts. It can generate full sentence questions, multiple choice questions, or a mix of the
    two styles.

    To filter out low quality questions, questions are assigned a score and ranked once they have
    been generated. Only the top k questions will be returned. This behaviour can be turned off
    by setting use_evaluator=False.
    """

    def __init__(self) -> None:
        QG_PRETRAINED = "iarfmoose/t5-base-question-generator"
        self.ANSWER_TOKEN = "<answer>"
        self.CONTEXT_TOKEN = "<context>"
        self.SEQ_LENGTH = 512

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.qg_tokenizer = AutoTokenizer.from_pretrained(
            QG_PRETRAINED, use_fast=False)
        self.qg_model = AutoModelForSeq2SeqLM.from_pretrained(QG_PRETRAINED)
        self.qg_model.to(self.device)
        self.qg_model.eval()

        self.qa_evaluator = QAEvaluator()

    def generate(
        self,
        article: str,
        use_evaluator: bool = True,
        num_questions: bool = None,
        answer_style: str = "sentences"
    ) -> List:
        """Takes an article and generates a set of question and answer pairs. If use_evaluator
        is True then QA pairs will be ranked and filtered based on their quality. answer_style
        should selected from ["all", "sentences", "multiple_choice"].
        """

        print("Generating questions...\n")

        qg_inputs, qg_answers = self.generate_qg_inputs(article, answer_style)
        generated_questions = self.generate_questions_from_inputs(qg_inputs)

        message = "{} questions doesn't match {} answers".format(
            len(generated_questions), len(qg_answers)
        )
        assert len(generated_questions) == len(qg_answers), message

        if use_evaluator:
            print("Evaluating QA pairs...\n")
            encoded_qa_pairs = self.qa_evaluator.encode_qa_pairs(
                generated_questions, qg_answers
            )
            scores = self.qa_evaluator.get_scores(encoded_qa_pairs)

            if num_questions:
                qa_list = self._get_ranked_qa_pairs(
                    generated_questions, qg_answers, scores, num_questions
                )
            else:
                qa_list = self._get_ranked_qa_pairs(
                    generated_questions, qg_answers, scores
                )

        else:
            print("Skipping evaluation step.\n")
            qa_list = self._get_all_qa_pairs(generated_questions, qg_answers)

        return qa_list

    def generate_qg_inputs(self, text: str, answer_style: str) -> Tuple[List[str], List[str]]:
        """Given a text, returns a list of model inputs and a list of corresponding answers.
        Model inputs take the form "answer_token <answer text> context_token <context text>" where
        the answer is a string extracted from the text, and the context is the wider text surrounding
        the context.
        """

        VALID_ANSWER_STYLES = ["all", "sentences", "multiple_choice"]

        if answer_style not in VALID_ANSWER_STYLES:
            raise ValueError(
                "Invalid answer style {}. Please choose from {}".format(
                    answer_style, VALID_ANSWER_STYLES
                )
            )

        inputs = []
        answers = []

        if answer_style == "sentences" or answer_style == "all":
            segments = self._split_into_segments(text)

            for segment in segments:
                sentences = self._split_text(segment)
                prepped_inputs, prepped_answers = self._prepare_qg_inputs(
                    sentences, segment
                )
                inputs.extend(prepped_inputs)
                answers.extend(prepped_answers)

        if answer_style == "multiple_choice" or answer_style == "all":
            sentences = self._split_text(text)
            prepped_inputs, prepped_answers = self._prepare_qg_inputs_MC(
                sentences
            )
            inputs.extend(prepped_inputs)
            answers.extend(prepped_answers)

        return inputs, answers

    def generate_questions_from_inputs(self, qg_inputs: List) -> List[str]:
        """Given a list of concatenated answers and contexts, with the form:
        "answer_token <answer text> context_token <context text>", generates a list of 
        questions.
        """
        generated_questions = []

        for qg_input in qg_inputs:
            question = self._generate_question(qg_input)
            generated_questions.append(question)

        return generated_questions

    def _split_text(self, text: str) -> List[str]:
        """Splits the text into sentences, and attempts to split or truncate long sentences."""
        MAX_SENTENCE_LEN = 128
        sentences = re.findall(".*?[.!\?]", text)
        cut_sentences = []

        for sentence in sentences:
            if len(sentence) > MAX_SENTENCE_LEN:
                cut_sentences.extend(re.split("[,;:)]", sentence))

        # remove useless post-quote sentence fragments
        cut_sentences = [s for s in sentences if len(s.split(" ")) > 5]
        sentences = sentences + cut_sentences

        return list(set([s.strip(" ") for s in sentences]))

    def _split_into_segments(self, text: str) -> List[str]:
        """Splits a long text into segments short enough to be input into the transformer network.
        Segments are used as context for question generation.
        """
        MAX_TOKENS = 490
        paragraphs = text.split("\n")
        tokenized_paragraphs = [
            self.qg_tokenizer(p)["input_ids"] for p in paragraphs if len(p) > 0
        ]
        segments = []

        while len(tokenized_paragraphs) > 0:
            segment = []

            while len(segment) < MAX_TOKENS and len(tokenized_paragraphs) > 0:
                paragraph = tokenized_paragraphs.pop(0)
                segment.extend(paragraph)
            segments.append(segment)

        return [self.qg_tokenizer.decode(s, skip_special_tokens=True) for s in segments]

    def _prepare_qg_inputs(
        self,
        sentences: List[str],
        text: str
    ) -> Tuple[List[str], List[str]]:
        """Uses sentences as answers and the text as context. Returns a tuple of (model inputs, answers).
        Model inputs are "answer_token <answer text> context_token <context text>" 
        """
        inputs = []
        answers = []

        for sentence in sentences:
            qg_input = f"{self.ANSWER_TOKEN} {sentence} {self.CONTEXT_TOKEN} {text}"
            inputs.append(qg_input)
            answers.append(sentence)

        return inputs, answers

    def _prepare_qg_inputs_MC(self, sentences: List[str]) -> Tuple[List[str], List[str]]:
        """Performs NER on the text, and uses extracted entities are candidate answers for multiple-choice
        questions. Sentences are used as context, and entities as answers. Returns a tuple of (model inputs, answers). 
        Model inputs are "answer_token <answer text> context_token <context text>"
        """
        spacy_nlp = en_core_web_sm.load()
        docs = list(spacy_nlp.pipe(sentences, disable=["parser"]))
        inputs_from_text = []
        answers_from_text = []

        for doc, sentence in zip(docs, sentences):
            entities = doc.ents
            if entities:

                for entity in entities:
                    qg_input = f"{self.ANSWER_TOKEN} {entity} {self.CONTEXT_TOKEN} {sentence}"
                    answers = self._get_MC_answers(entity, docs)
                    inputs_from_text.append(qg_input)
                    answers_from_text.append(answers)

        return inputs_from_text, answers_from_text

    def _get_MC_answers(self, correct_answer: Any, docs: Any) -> List[Mapping[str, Any]]:
        """Finds a set of alternative answers for a multiple-choice question. Will attempt to find
        alternatives of the same entity type as correct_answer if possible.
        """
        entities = []

        for doc in docs:
            entities.extend([{"text": e.text, "label_": e.label_}
                            for e in doc.ents])

        # remove duplicate elements
        entities_json = [json.dumps(kv) for kv in entities]
        pool = set(entities_json)
        num_choices = (
            min(4, len(pool)) - 1
        )  # -1 because we already have the correct answer

        # add the correct answer
        final_choices = []
        correct_label = correct_answer.label_
        final_choices.append({"answer": correct_answer.text, "correct": True})
        pool.remove(
            json.dumps({"text": correct_answer.text,
                       "label_": correct_answer.label_})
        )

        # find answers with the same NER label
        matches = [e for e in pool if correct_label in e]

        # if we don't have enough then add some other random answers
        if len(matches) < num_choices:
            choices = matches
            pool = pool.difference(set(choices))
            choices.extend(random.sample(pool, num_choices - len(choices)))
        else:
            choices = random.sample(matches, num_choices)

        choices = [json.loads(s) for s in choices]

        for choice in choices:
            final_choices.append({"answer": choice["text"], "correct": False})

        random.shuffle(final_choices)
        return final_choices

    @torch.no_grad()
    def _generate_question(self, qg_input: str) -> str:
        """Takes qg_input which is the concatenated answer and context, and uses it to generate
        a question sentence. The generated question is decoded and then returned.
        """
        encoded_input = self._encode_qg_input(qg_input)
        output = self.qg_model.generate(input_ids=encoded_input["input_ids"])
        question = self.qg_tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )
        return question

    def _encode_qg_input(self, qg_input: str) -> torch.tensor:
        """Tokenizes a string and returns a tensor of input ids corresponding to indices of tokens in 
        the vocab.
        """
        return self.qg_tokenizer(
            qg_input,
            padding='max_length',
            max_length=self.SEQ_LENGTH,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

    def _get_ranked_qa_pairs(
        self, generated_questions: List[str], qg_answers: List[str], scores, num_questions: int = 10
    ) -> List[Mapping[str, str]]:
        """Ranks generated questions according to scores, and returns the top num_questions examples.
        """
        if num_questions > len(scores):
            num_questions = len(scores)
            print((
                f"\nWas only able to generate {num_questions} questions.",
                "For more questions, please input a longer text.")
            )

        qa_list = []

        for i in range(num_questions):
            index = scores[i]
            qa = {
                "question": generated_questions[index].split("?")[0] + "?",
                "answer": qg_answers[index]
            }
            qa_list.append(qa)

        return qa_list

    def _get_all_qa_pairs(self, generated_questions: List[str], qg_answers: List[str]):
        """Formats question and answer pairs without ranking or filtering."""
        qa_list = []

        for question, answer in zip(generated_questions, qg_answers):
            qa = {
                "question": question.split("?")[0] + "?",
                "answer": answer
            }
            qa_list.append(qa)

        return qa_list


class QAEvaluator:
    """Wrapper for a transformer model which evaluates the quality of question-answer pairs.
    Given a QA pair, the model will generate a score. Scores can be used to rank and filter
    QA pairs.
    """

    def __init__(self) -> None:

        QAE_PRETRAINED = "iarfmoose/bert-base-cased-qa-evaluator"
        self.SEQ_LENGTH = 512

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.qae_tokenizer = AutoTokenizer.from_pretrained(QAE_PRETRAINED)
        self.qae_model = AutoModelForSequenceClassification.from_pretrained(
            QAE_PRETRAINED
        )
        self.qae_model.to(self.device)
        self.qae_model.eval()

    def encode_qa_pairs(self, questions: List[str], answers: List[str]) -> List[torch.tensor]:
        """Takes a list of questions and a list of answers and encodes them as a list of tensors."""
        encoded_pairs = []

        for question, answer in zip(questions, answers):
            encoded_qa = self._encode_qa(question, answer)
            encoded_pairs.append(encoded_qa.to(self.device))

        return encoded_pairs

    def get_scores(self, encoded_qa_pairs: List[torch.tensor]) -> List[float]:
        """Generates scores for a list of encoded QA pairs."""
        scores = {}

        for i in range(len(encoded_qa_pairs)):
            scores[i] = self._evaluate_qa(encoded_qa_pairs[i])

        return [
            k for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)
        ]

    def _encode_qa(self, question: str, answer: str) -> torch.tensor:
        """Concatenates a question and answer, and then tokenizes them. Returns a tensor of 
        input ids corresponding to indices in the vocab.
        """
        if type(answer) is list:
            for a in answer:
                if a["correct"]:
                    correct_answer = a["answer"]
        else:
            correct_answer = answer

        return self.qae_tokenizer(
            text=question,
            text_pair=correct_answer,
            padding="max_length",
            max_length=self.SEQ_LENGTH,
            truncation=True,
            return_tensors="pt",
        )

    @torch.no_grad()
    def _evaluate_qa(self, encoded_qa_pair: torch.tensor) -> float:
        """Takes an encoded QA pair and returns a score."""
        output = self.qae_model(**encoded_qa_pair)
        return output[0][0][1]


class GapFillQAGenerator:
    def __init__(self, data, noOfQues):
        self.summary = str(data)
        self.noOfQues = int(noOfQues)

    def get_trivial_sentences(self):
        sentences = nltk.sent_tokenize(self.summary)
        trivial_sentences = list()
        for sent in sentences:
            trivial = self.identify_trivial_sentences(sent)
            if trivial:
                trivial_sentences.append(trivial)
            else:
                continue
        return trivial_sentences

    def identify_trivial_sentences(self, sentence):
        tags = nltk.pos_tag(sentence)
        if tags[0][1] == "RB" or len(nltk.word_tokenize(sentence)) < 4:
            return None

        noun_phrases = list()
        grammer = r"""
            CHUNK: {<NN>+<IN|DT>*<NN>+}
                {<NN>+<IN|DT>*<NNP>+}
                {<NNP>+<NNS>*}
            """
        chunker = nltk.RegexpParser(grammer)
        tokens = nltk.word_tokenize(sentence)
        pos_tokens = nltk.tag.pos_tag(tokens)
        tree = chunker.parse(pos_tokens)

        for subtree in tree.subtrees():
            if subtree.label() == "CHUNK":
                temp = ""
                for sub in subtree:
                    temp += sub[0]
                    temp += " "
                temp = temp.strip()
                noun_phrases.append(temp)

        replace_nouns = []
        for word, _ in tags:
            for phrase in noun_phrases:
                if phrase[0] == '\'':
                    break
                if word in phrase:
                    [replace_nouns.append(phrase_word)
                     for phrase_word in phrase.split()[-2:]]
                    break
            if len(replace_nouns) == 0:
                replace_nouns.append(word)
            break

        if len(replace_nouns) == 0:
            return None

        val = 99
        for i in replace_nouns:
            if len(i) < val:
                val = len(i)
            else:
                continue

        trivial = {
            "Answer": " ".join(replace_nouns),
            "Key": val
        }

        if len(replace_nouns) == 1:
            trivial["Similar"] = self.answer_options(replace_nouns[0])
        else:
            trivial["Similar"] = []

        replace_phrase = " ".join(replace_nouns)
        blanks_phrase = ("__________" * len(replace_nouns)).strip()
        expression = re.compile(re.escape(replace_phrase), re.IGNORECASE)
        sentence = expression.sub(blanks_phrase, str(sentence), count=1)
        trivial["Question"] = sentence
        return trivial

    @staticmethod
    def answer_options(word):
        synsets = wn.synsets(word, pos="n")

        if len(synsets) == 0:
            return []
        else:
            synset = synsets[0]

        hypernym = synset.hypernyms()[0]
        hyponyms = hypernym.hyponyms()
        similar_words = []
        for hyponym in hyponyms:
            similar_word = hyponym.lemmas()[0].name().replace("_", " ")
            if similar_word != word:
                similar_words.append(similar_word)
            if len(similar_words) == 8:
                # print(similar_words)
                break
        return similar_words

    def generate_test(self):
        trivial_pair = self.get_trivial_sentences()
        question_answer = list()
        for que_ans_dict in trivial_pair:
            if que_ans_dict["Key"] > int(self.noOfQues):
                question_answer.append(que_ans_dict)
            else:
                continue
        question = list()
        answer = list()
        gap_qns_ans = list()
        while len(question) < int(self.noOfQues):
            # print("genratting",len(self.noOfQues))
            rand_num = np.random.randint(0, len(question_answer))
            if question_answer[rand_num]["Question"] not in question:
                question.append(question_answer[rand_num]["Question"])
                answer.append(question_answer[rand_num]["Answer"])
                gap_qns_ans.append(
                    {'question': question_answer[rand_num]["Question"], 'answer': question_answer[rand_num]["Answer"]})
            else:
                continue
        return gap_qns_ans


class BoolQAGenerator:
    def __init__(self):
        self.set_seed(42)
        self.model = T5ForConditionalGeneration.from_pretrained(
            'ramsrigouthamg/t5_boolean_questions')
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def set_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def beam_search_decoding(self, inp_ids, attn_mask, no_of_qns):
        beam_output = self.model.generate(input_ids=inp_ids,
                                          attention_mask=attn_mask,
                                          max_length=256,
                                          num_beams=10,
                                          num_return_sequences=no_of_qns,
                                          no_repeat_ngram_size=2,
                                          early_stopping=True
                                          )
        Questions = [self.tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True) for out in
                     beam_output]
        return [str(Question).strip().capitalize() for Question in Questions]

    def get_bool_qnsans(self, document, no_of_qns=5):
        encoding = self.tokenizer.encode_plus(
            document, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"].to(
            self.device), encoding["attention_mask"].to(self.device)
        output = self.beam_search_decoding(input_ids, attention_masks, no_of_qns)
        bool_qns_ans = list()
        for question in output:
            ans = bq.predict(document, question)
            ans = 'Yes' if ans else 'No'
            bool_qns_ans.append({'question': question, 'answer': ans})
        return bool_qns_ans


def print_qa(qa_list,isSingle=False):
    wh_qns_ans = list()
    mcq_qns_ans = list()
    subj_qns_ans = list()
    for qa_pair in qa_list:
        if isinstance(qa_pair['answer'], list):
            qoa = dict()
            opList = list()
            qoa['question'] = qa_pair['question']
            for ans in qa_pair['answer']:
                if (isSingle == False):
                    opList.append(ans['answer'])
                if (ans['correct'] == True):
                    qoa['answer'] = ans['answer']
            if (isSingle == False):
                qoa['options'] = opList
                mcq_qns_ans.append(qoa)
            else:
                wh_qns_ans.append(qoa)
        else:
            qa = dict()
            qa['question'] = qa_pair['question']
            qa['answer'] = qa_pair["answer"]
            subj_qns_ans.append(qa)
    if isinstance(qa_list[0]['answer'], list):
        if isSingle == False:
            return mcq_qns_ans
        return wh_qns_ans
    return subj_qns_ans
