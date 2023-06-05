from typing import List

class IntentRecognitionForDialog():
    """
    Prompt class for `Intent Recognition` task (for `bot-user` dialogs)
    """
    def __init__(self, fallback_id: int, fallback_value: str) -> None:
        self.group = 0
        self.fallback_id = fallback_id
        self.fallback_value = fallback_value

    def build_prompt(self, bot_question: str, user_answer: str, candidate_labels: List[str], label: int = None, *args, **kwargs):
        res_list = []

        for candidate in candidate_labels:
            input_text = 'premise: question: {} answer: {} claim: The answer to the question is similar to: {} hypothesis: The premise entails the claim.'.format(
                bot_question, user_answer, convert_exemple(candidate)
            )
            target_text = ''
            label_name = candidate_labels[label] if label != self.fallback_id else self.fallback_value

            res_list.append(dict(
                input_text=input_text,
                target_text=target_text,
                label=candidate_labels.index(label_name) if label_name in candidate_labels else self.fallback_id,
                label_name=label_name,
                candidate_labels=candidate_labels,
                group=self.group
            ))
        
        self.group += 1

        return res_list

class BoolQA():
    """
    Prompt class for `Boolean Question Answering` task
    """
    def __init__(self, fallback_id: int, fallback_value: str) -> None:
        self.group = 0
        self.fallback_id = fallback_id
        self.fallback_value = fallback_value

    def build_prompt(self, question: str, answer: str, candidate_labels: List[str], label: int = None, *args, **kwargs):
        res_list = []

        for candidate in candidate_labels:
            input_text = 'premise: question: {} answer: {} claim: The answer to the question means {} hypothesis: The premise entails the claim.'.format(
                question, answer, convert_exemple(candidate)
            )
            target_text = ''
            label_name = candidate_labels[label] if label != self.fallback_id else self.fallback_value

            res_list.append(dict(
                input_text=input_text,
                target_text=target_text,
                label=candidate_labels.index(label_name) if label_name in candidate_labels else self.fallback_id,
                label_name=label_name,
                candidate_labels=candidate_labels,
                group=self.group
            ))
        
        self.group += 1

        return res_list

class SentimentAnalysisForDialog():
    """
    Prompt class for `Sentiment Analysis` task (for `bot-user` dialogs)
    """
    def __init__(self) -> None:
        self.group = 0

    def build_prompt(self, bot_question: str, user_answer: str, candidate_labels: List[str], label: int = None, *args, **kwargs):
        res_list = []

        for candidate in candidate_labels:
            input_text = 'premise: question: {} answer: {} claim: The answer to the question expresses a sentiment of {} hypothesis: The premise entails the claim.'.format(
                bot_question, user_answer, convert_exemple(candidate)
            )
            target_text = ''
            res_list.append(dict(
                input_text=input_text,
                target_text=target_text,
                label=label,
                hypothesis_classes=candidate_labels,
                group=self.group
            ))
        
        self.group += 1

        return res_list

class GlobalBoolQA():
    """
    Prompt class for `Boolean Question Answering` task
    """
    def __init__(self, fallback_id: int, fallback_value: str) -> None:
        self.group = 0
        self.fallback_id = fallback_id
        self.fallback_value = fallback_value

    def build_prompt(self, question: str, passage: str, candidate_labels: List[str], label: int = None, *args, **kwargs):
        res_list = []

        for candidate in candidate_labels:
            input_text = 'premise: question: {}? passage: {} claim: Based on this passage, the answer to the question is {}. hypothesis: The premise entails the claim.'.format(
                question, passage, 'yes' if candidate == 'true' else 'no'
            )
            target_text = ''
            label_name = candidate_labels[label] if label != self.fallback_id else self.fallback_value

            res_list.append(dict(
                input_text=input_text,
                target_text=target_text,
                label=candidate_labels.index(label_name) if label_name in candidate_labels else self.fallback_id,
                label_name=label_name,
                candidate_labels=candidate_labels,
                group=self.group
            ))
        
        self.group += 1

        return res_list

class SentimentAnalysis():
    """
    Prompt class for `Sentiment Analysis` task
    """
    def __init__(self, fallback_id: int, fallback_value: str) -> None:
        self.group = 0
        self.fallback_id = fallback_id
        self.fallback_value = fallback_value

    def build_prompt(self, document: str, candidate_labels: List[str], label: int = None, *args, **kwargs):
        res_list = []

        for candidate in candidate_labels:
            input_text = 'premise: document: {} claim: This document expresses a sentiment of {} hypothesis: The premise entails the claim.'.format(
                document, convert_exemple(candidate)
            )
            target_text = ''
            label_name = candidate_labels[label] if label != self.fallback_id else self.fallback_value

            res_list.append(dict(
                input_text=input_text,
                target_text=target_text,
                label=candidate_labels.index(label_name) if label_name in candidate_labels else self.fallback_id,
                label_name=label_name,
                candidate_labels=candidate_labels,
                group=self.group
            ))
        
        self.group += 1

        return res_list

class NaturalLanguageInference():
    """
    Prompt class for `Natural Language Inference` task
    """
    def __init__(self, fallback_id: int, fallback_value: str) -> None:
        self.group = 0
        self.fallback_id = fallback_id
        self.fallback_value = fallback_value

    def build_prompt(self, premise: str, claim: str, candidate_labels: List[str], label: int = None, *args, **kwargs):
        res_list = []

        for candidate in candidate_labels:
            input_text = 'premise: {} claim: {} hypothesis: The premise {} the claim'.format(
                premise, claim, convert_exemple(candidate)
            )
            target_text = ''
            label_name = candidate_labels[label] if label != self.fallback_id else self.fallback_value

            res_list.append(dict(
                input_text=input_text,
                target_text=target_text,
                label=candidate_labels.index(label_name) if label_name in candidate_labels else self.fallback_id,
                label_name=label_name,
                candidate_labels=candidate_labels,
                group=self.group
            ))
        
        self.group += 1

        return res_list

class Paraphrase():
    """
    Prompt class for `Paraphrase` task
    """
    def __init__(self, fallback_id: int, fallback_value: str) -> None:
        self.group = 0
        self.fallback_id = fallback_id
        self.fallback_value = fallback_value

    def build_prompt(self, document1: str, document2: str, candidate_labels: List[str], label: int = None, *args, **kwargs):
        res_list = []

        for candidate in candidate_labels:
            input_text = 'document1: {} document2: {} hypothesis: document1 and document2 are {}'.format(
                document1, document2, convert_exemple(candidate)
            )
            target_text = ''
            label_name = candidate_labels[label] if label != self.fallback_id else self.fallback_value

            res_list.append(dict(
                input_text=input_text,
                target_text=target_text,
                label=candidate_labels.index(label_name) if label_name in candidate_labels else self.fallback_id,
                label_name=label_name,
                candidate_labels=candidate_labels,
                group=self.group
            ))
        
        self.group += 1

        return res_list


def convert_exemple(name: str) -> str:
    """"
    Convert the intent name to a natural language sentence. Create an example.
    
    Params:
        - name: name of the intent to convert
    
    Returns:
        - converted intent
    """
    if name == 'yes':
        new_name = 'yes'
    elif name == 'no':
        new_name = 'no'
    
    elif name == 'positive':
        new_name = 'positivity'
    elif name == 'neutral':
        new_name = 'neutrality'
    elif name == 'negative':
        new_name = 'negativity'

    elif name == 'entailment':
        new_name = 'entails'
    elif name == 'neutral-nli':
        new_name = 'is neutral to'
    elif name == 'contradiction':
        new_name = 'contradicts'
    elif name == 'not-entailment':
        new_name = 'does not entail'
    
    elif name == 'duplicate':
        new_name = 'semantically equivalent'
    elif name == 'not-duplicate':
        new_name = 'not sementically equivalent'

    elif name == 'favorite-continent-asia':
        new_name = 'my favorite continent is Asia'
    elif name == 'favorite-continent-australia':
        new_name = 'my favorite continent is Australia'
    elif name == 'favorite-continent-africa':
        new_name = 'my favorite continent is Africa'
    elif name == 'favorite-continent-antarctica':
        new_name = 'my favorite continent is Antartica'
    elif name == 'favorite-continent-north-america':
        new_name = 'my favorite continent is North America'
    elif name == 'favorite-continent-europe':
        new_name = 'my favorite continent is Europe'
    elif name == 'favorite-continent-south-america':
        new_name = 'my favorite continent is South America'
    
    elif name == 'likes-tennis':
        new_name = 'my favorite sport is tennis'
    elif name == 'likes-baseball':
        new_name = 'my favorite sport is baseball'
    elif name == 'likes-basketball':
        new_name = 'my favorite sport is basketball'
    
    elif name == 'topic-books-likes-both-genre':
        new_name = 'I like reading both genre'
    elif name == 'topic-books-likes-fiction':
        new_name = 'I like reading fiction'
    elif name == 'topic-books-likes-non-fiction':
        new_name = 'I like reading non-fiction'

    elif name == 'topic-food-for-dinner':
        new_name = 'it is consumed for dinner'
    elif name == 'topic-food-for-breakfast':
        new_name = 'it is consumed for breakfast'
    elif name == 'topic-food-for-lunch':
        new_name = 'it is consumed for lunch'

    elif name == 'games':
        new_name = 'I like playing games'
    elif name == 'gardening':
        new_name = 'I like gardening'
    elif name == 'working-out':
        new_name = 'I like working-out'

    elif name == 'topic-hometown-big-city':
        new_name = 'it is a big city'
    elif name == 'topic-hometown-small-city':
        new_name = 'it is a small city'

    elif name == 'topic-profession-generic-profession':
        new_name = 'I have a generic profession'
    elif name == 'topic-profession-evil-profession':
        new_name = 'I have a profession related to evil'
    elif name == 'student':
        new_name = 'I am a student'

    elif name == 'topic-speaker-age-less-than-18-answer':
        new_name = 'I am still a child, less than 18 years old'
    elif name == 'topic-speaker-age-greater-than-18-answer':
        new_name = 'I am an adult, more than 18 years old'

    elif name == 'topic-travel-homecountry-human-from-india':
        new_name = 'I am from India'
    elif name == 'topic-travel-homecountry-human-from-japan':
        new_name = 'I am from Japan'
    elif name == 'topic-travel-homecountry-sarcastic-location':
        new_name = 'I am from a weird place'
    elif name == 'topic-travel-homecountry-human-from-usa':
        new_name = 'I am from the USA'
    elif name == 'topic-travel-homecountry-human-from-china':
        new_name = 'I am from China'

    elif name == 'favorite-season-summer':
        new_name = 'my favorite season is summer'
    elif name == 'favorite-season-winter':
        new_name = 'my favorite season is winter'
    elif name == 'favorite-season-spring':
        new_name = 'my favorite season is spring'
    elif name == 'favorite-season-autumn':
        new_name = 'my favorite season is autumn'

    elif name == 'watch-in-person':
        new_name = 'I like to watch it in person, where the action is'
    elif name == 'watches-on-tv':
        new_name = 'I like to watch it on TV'

    elif name == 'natural-wonders':
        new_name = 'I prefer natural wonders'
    elif name == 'man-made-monuments-answer':
        new_name = 'I prefer man-made monuments'

    elif name == 'topic-books-physical-books':
        new_name = 'I buy physical books wich are paper books'
    elif name == 'topic-books-ebooks':
        new_name = 'I buy ebooks which are digital books'

    elif name == 'topic-books-most-sold-book-rowling':
        new_name = 'J. K. Rowling sold more books'
    elif name == 'topic-books-most-sold-book-tolkien':
        new_name = 'J. R. R. Tolkien sold more books'

    elif name == 'topic-hometown-type-of-building-apartment-answer':
        new_name = 'I live in a appartment'
    elif name == 'topic-hometown-type-of-building-house-answer':
        new_name = 'I live in a house'

    elif name == 'topic-pet-eat-answer-little':
        new_name = 'my pet eats a little'
    elif name == 'topic-pet-eat-answer-lots':
        new_name = 'my pet eats a lot'

    elif name == 'topic-travel-homecountry-favorite-hemisphere-north':
        new_name = 'my favorite side of the globe is the north'
    elif name == 'topic-travel-homecountry-favorite-hemisphere-south':
        new_name = 'my favorite side of the globe is the south'

    elif name == 'like-to-play-sports':
        new_name = 'I like to play sports'
    elif name == 'like-to-watch-sports':
        new_name = 'I like to watch sports'

    elif name == 'topic-day-one-session-one-age-wrappingup-adulthood':
        new_name = 'I prefer adulthood'
    elif name == 'topic-day-one-session-one-age-wrappingup-childhood':
        new_name = 'I prefer chilhood'
    elif name == 'topic-day-one-session-one-age-wrappingup-oldage':
        new_name = 'I prefer old age'

    elif name == 'topic-language-learn-english-at-school':
        new_name = 'learn it at school'
    elif name == 'topic-language-learn-english-at-home':
        new_name = 'learn it at home'

    elif name == 'topic-birthday-days-february':
        new_name = '28 (twenty-eight) or 29 (twenty-nine) days'
    elif name == 'topic-birthday-days-thirty':
        new_name = '30 (thirty) days'
    elif name == 'topic-birthday-days-thirtyone':
        new_name = '31 (thirty-one) days'

    elif name == 'topic-day-three-food-noodles':
        new_name = 'I like noodles'
    elif name == 'topic-day-three-food-burgers':
        new_name = 'I like burgers'
    elif name == 'topic-day-three-food-pizza':
        new_name = 'I like pizza'

    elif name == 'topic-day-three-number-meals-between-three-six':
        new_name = 'between 3 and 6'
    elif name == 'topic-day-three-number-meals-greaterthan-six':
        new_name = 'greater than 6'
    elif name == 'topic-day-three-number-meals-lessthan-three':
        new_name = 'less than 3'

    elif name == 'likes-to-play-sports':
        new_name = 'I like to play'
    elif name == 'likes-to-watch-sports-or-fallback':
        new_name = 'I like to watch'

    elif name == 'topic-day-four-school-favorite-subject-science':
        new_name = 'my favorite subject is science'
    elif name == 'topic-day-four-school-favorite-subject-social':
        new_name = 'my favorite subject is socials'
    elif name == 'topic-day-four-school-favorite-subject-math':
        new_name = 'my favorite subject is mathematics'
    elif name == 'topic-day-four-school-favorite-subject-english':
        new_name = 'my favorite subject is english'
    
    elif name == 'topic-day-four-school-extra-curriculars-music':
        new_name = 'after shcool I play music'
    elif name == 'topic-day-four-school-extra-curriculars-sports':
        new_name = 'after school I play sports'
    
    elif name == 'topic-day-five-weather-rain':
        new_name = 'the rain is better'
    elif name == 'topic-day-five-weather-sun':
        new_name = 'the sun is better'

    elif name == 'topic-day-five-weather-favorite-season-summer':
        new_name = 'my favorite season is summer'
    elif name == 'topic-day-five-weather-favorite-season-winter':
        new_name = 'my favorite season is winter'
    elif name == 'topic-day-five-weather-favorite-season-fall':
        new_name = 'my favorite season is fall'
    elif name == 'topic-day-five-weather-favorite-season-spring':
        new_name = 'my favorite season is spring'

    elif name == 'topic-day-five-travel-sightseeing':
        new_name = 'I prefer sightseeing'
    elif name == 'topic-day-five-travel-food':
        new_name = 'I prefer the food'

    elif name == 'topic-olympics-select-user-would-compete-volleyball':
        new_name = 'I would compete in volleball'
    elif name == 'topic-olympics-select-user-would-compete-tennis':
        new_name = 'I would compete in tennis'
    elif name == 'topic-olympics-select-user-would-compete-diving':
        new_name = 'I would compete in diving'
    elif name == 'topic-olympics-select-user-would-compete-archery':
        new_name = 'I would compete in archery'

    elif name == 'topic-olympics-user-height-above-seventytwo':
        new_name = 'I am taller than 72 (seventy-two) inches'
    elif name == 'topic-olympics-user-height-below-seventytwo':
        new_name = 'I am less tall than 72 (seventy-two) inches'

    elif name == 'topic-olympics-haru-height-above-twentyfour':
        new_name = 'you are taller than 24 (twenty-four) inches'
    elif name == 'topic-olympics-haru-height-below-twentyfour':
        new_name = 'you are less tall than 24 (twenty-four) inches'

    else:
        print('`{}` not converted'.format(name))
        new_name = name

    return new_name