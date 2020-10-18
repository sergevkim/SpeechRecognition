from torch import Tensor


class TokenConverter:
    def __init__(self, symbols=list(' abcdefghijklmnopqrstuvwxyz')):
        self.symbols = symbols
        self.symbol2number = dict()
        self.number2symbol = dict()

        for number, symbol in enumerate(symbols):
            self.symbol2number[symbol] = number
            self.number2symbol[number] = symbol

    def text_to_numbers(self, symbols):
        result = Tensor([self.symbol2number[symbol] for symbol in symbols])

        return result

    def numbers_to_text(self, numbers):
        result = ''.join(self.number2symbol for number in numbers)

        return result

