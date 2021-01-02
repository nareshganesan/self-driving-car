import json
import numpy as np
from tensorflow.python.framework.errors import InvalidArgumentError

def get_result(student_func):
    
    """
    Run unit tests against <student_func>
    """

    answer = 123
    result = {
        'correct': False,
        'feedback': 'That\'s the wrong answer.  It should print {}'.format(answer),
        'comment': ''}

    try:
        output = student_func()
        if not output:
            result['feedback'] = 'No output found'
        if not isinstance(output, np.ndarray):
            result['feedback'] = 'Output is the wrong type.'
            result['comment'] = 'The output should come from running the session.'
        if output == answer:
            result['correct'] = True
            result['feedback'] = 'You got it right.  You figured out how to use feed_dict!'
    except InvalidArgumentError as err:
        if err.message.startswith('You must feed a value for placeholder tensor'):
            result['feedback'] = 'The placeholder is not being set.'
            result['comment'] = 'Try using the feed_dict.'
    except Exception as err:
        result['feedback'] = 'Something went wrong with your submission:'
        result['comment'] = str(err)
    
    print("{} \n{}".format(result.get('feedback'), result.get('comment')))
    

def get_result1(student_output):
    """
    Run unit tests against <student_func>
    """
    
    answer = 4.0
    result = {
        'correct': False,
        'feedback': 'That\'s the wrong answer.  It should print {}'.format(answer),
        'comment': ''}

    try:
        student_output = np.float32(student_output)
        if not student_output:
            result['feedback'] = 'No output found'
        elif student_output == answer:
            result['correct'] = True
            result['feedback'] = 'That\'s right!  You correctly turned the algorithm to TensorFlow'
    except TypeError as err:
        if str(err).endswith('into a Tensor or Operation.)'):
            result['feedback'] = 'TensorFlow session requires a tensor to run'
        else:
            raise

    return result


def run_grader(student_output):

    try:
        # Get grade result information
        result = get_result1(student_output)
    except Exception as err:
        # Default error result
        result = {
            'correct': False,
            'feedback': 'Something went wrong with your submission:',
            'comment': str(err)}
        
    feedback = result.get('feedback')
    comment = result.get('comment')

    print(f"{feedback}\n {comment}\n")