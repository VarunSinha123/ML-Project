import sys
from src.logger import logging
def error_message_details(error, error_details: tuple):
    _, _, exc_tb = error_details
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        exc_tb.tb_frame.f_code.co_filename, exc_tb.tb_lineno, str(error))
    return error_message

class CustomException(Exception):
    def __init__(self, error_message: str, error_details: tuple):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_details)
        
    def __str__(self):
        return self.error_message
