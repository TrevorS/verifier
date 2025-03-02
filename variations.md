# Monetary Expression Variations

This document lists various monetary expression formats, focusing on business and check writing conventions.  It excludes duplicates from the original prompt and reorganizes the list for clarity.

## Helper Functions

These are the reusable building blocks for creating monetary expressions:

1.  **`dollars_to_words(dollars: int) -> str`**: Converts a dollar amount (integer) to its word representation.
    *   Example: `25` → `"twenty-five"`
    *   Example: `2500000` → `"two million five hundred thousand"`

2.  **`cents_to_words(cents: int) -> str`**: Converts a cent amount (integer) to its word representation.
    *   Example: `50` → `"fifty"`

3.  **`dollars_to_text(dollars: int) -> str`**: Formats a dollar amount with the correct pluralization of "dollar".
    *   Example: `1` → `"one dollar"`, `25` → `"twenty-five dollars"`
    *   Example: `1000000` → `"one million dollars"`

4.  **`cents_to_text(cents: int) -> str`**: Formats a cent amount with the correct pluralization of "cent".
    *   Example: `1` → `"one cent"`, `50` → `"fifty cents"`

5.  **`number_with_and(number: int) -> str`**:  Formats numbers over 100, inserting "and" appropriately.
    *   Example: `125` → `"one hundred and twenty-five"`
    *   Example: `1250000` → `"one million, two hundred and fifty thousand"`

6.  **`number_without_and(number: int) -> str`**: Formats numbers over 100, *omitting* "and".
    *   Example: `125` → `"one hundred twenty-five"`
    *   Example: `1250000` → `"one million, two hundred fifty thousand"`

7.  **`unhyphenate_compound(text: str) -> str`**: Removes hyphens from compound numbers.
    *   Example: `"twenty-five"` → `"twenty five"`

8.  **`get_denominations(cents: int) -> dict`**: Breaks down a cent amount into its constituent denominations.
    *   Example: `85` → `{"quarters": 3, "dimes": 1, "nickels": 0, "pennies": 0}`

9.  **`fraction_to_words(numerator: int, denominator: int) -> str`**: Converts a fraction to its word representation.
    *   Example: `(1, 2)` → `"one half"`, `(3, 4)` → `"three quarters"`

10. **`format_large_number(number: int, use_and: bool = True) -> str`**: Formats large numbers with commas and appropriate groupings, with option to use "and".
    *   Example: `2345678` → `"two million, three hundred and forty-five thousand, six hundred and seventy-eight"`

11. **`is_round_amount(amount: float) -> bool`**: Checks if an amount is a whole number of dollars.
    *   Example: `1000.00` → `True`, `1234.56` → `False`

12. **`is_business_check_amount(amount: float) -> bool`**:  A heuristic to guess if an amount is more likely to be from a business check.
    *   Example: `8432.17` → likely business, `50.00` → could be either

## Monetary Expression Variations

The following sections detail the different monetary expression variations, categorized for clarity.

### Standard and Business Check Variations

1.  **`standard_format(amount: float) -> str`**: Standard dollars and cents format.
    *   Example: `25.10` → `"twenty-five dollars and ten cents"`
    *   Example: `5423789.65` → `"five million, four hundred twenty-three thousand, seven hundred eighty-nine dollars and sixty-five cents"`

2.  **`fractional_cents_format(amount: float) -> str`**: Dollars with cents expressed as a fraction (xx/100).
    *   Example: `25.10` → `"twenty-five dollars and 10/100"`
    *   Example: `3245678.90` → `"three million, two hundred forty-five thousand, six hundred seventy-eight dollars and 90/100"`

3.  **`dollars_and_xx_100_format(amount: float) -> str`**:  Explicit "xx/100" for cents, even if zero.
    *   Example: `25.00` → `"twenty-five dollars and 00/100"`
    *   Example: `1500000.00` → `"one million, five hundred thousand dollars and 00/100"`

4.  **`dollars_after_fraction_format(amount: float) -> str`**: Fractional cents with "dollars" at the end.
    *   Example: `25.10` → `"twenty-five and 10/100 dollars"`
    *   Example: `2568432.75` → `"two million, five hundred sixty-eight thousand, four hundred thirty-two and 75/100 dollars"`

5.  **`only_dollars_format(amount: float) -> str`**:  "dollars only" for whole dollar amounts.
    *   Example: `25.00` → `"twenty-five dollars only"`
    *   Example: `7650000.00` → `"seven million, six hundred fifty thousand dollars only"`

6.  **`no_cents_specification(amount: float) -> str`**: Omits cents if the value is zero.
    *   Example: `25.00` → `"twenty-five dollars"`
    *   Example: `4580000.00` → `"four million, five hundred eighty thousand dollars"`

7.  **`exact_amount_format(amount: float) -> str`**: Includes "exact" or "exactly".
    *   Example: `8547.32` → `"eight thousand, five hundred forty-seven dollars and thirty-two cents exact"`

8. **`business_exactly_format(amount: float) -> str`**: Uses "exactly" at the beginning of the string.
    *   Example: `13579.80` → `"exactly thirteen thousand, five hundred seventy-nine dollars and eighty cents"`

9.  **`xx_100_dollars_format(amount: float) -> str`**:  Fractional cents *before* "dollars".
    *   Example: `2567.50` → `"two thousand five hundred sixty-seven and 50/100 dollars"`
    *   Example: `8765432.10` → `"eight million seven hundred sixty-five thousand four hundred thirty-two and 10/100 dollars"`

10. **`exact_amount_fraction_format(amount: float) -> str`**: Combines "exactly" with fractional cents.
    *   Example: `1234.75` → `"exactly one thousand two hundred thirty-four and 75/100 dollars"`
    *   Example: `5000000.25` → `"exactly five million and 25/100 dollars"`

11. **`and_no_100_format(amount: float) -> str`**: "And 00/100" for zero cents.
    * Example: `789.00` → `"seven hundred eighty-nine dollars and 00/100"`
    *   Example: `3670000.00` → `"three million six hundred seventy thousand dollars and 00/100"`

### Number Formation Variations

12. **`non_hyphenated_format(amount: float) -> str`**:  Compound numbers without hyphens.
    *   Example: `25.10` → `"twenty five dollars and ten cents"`
    *   Example: `2342567.89` → `"two million three hundred forty two thousand five hundred sixty seven dollars and eighty nine cents"`

13. **`with_and_in_numbers_format(amount: float) -> str`**: Includes "and" in numbers over 100.
    *   Example: `125.10` → `"one hundred and twenty-five dollars and ten cents"`
    *   Example: `4567890.12` → `"four million, five hundred and sixty-seven thousand, eight hundred and ninety dollars and twelve cents"`

14. **`no_and_in_numbers_format(amount: float) -> str`**: Omits "and" in numbers over 100.
    *   Example: `125.10` → `"one hundred twenty-five dollars and ten cents"`
    *   Example: `3456789.01` → `"three million, four hundred fifty-six thousand, seven hundred eighty-nine dollars and one cent"`

15. **`hundreds_as_multiples_format(amount: float) -> str`**: Expresses hundreds as multiples (e.g., "fifteen hundred").
    *   Example: `1500.00` → `"fifteen hundred dollars"`
    *   Example: `9900.00` → `"ninety-nine hundred dollars"`

16. **`comma_separated_format(amount: float) -> str`**:  Includes commas in the written number.
    *   Example: `1234567.89` → `"one million, two hundred thirty-four thousand, five hundred sixty-seven dollars and eighty-nine cents"`

17. **`no_commas_format(amount: float) -> str`**: Omits commas in the written number.
    *   Example: `1234567.89` → `"one million two hundred thirty-four thousand five hundred sixty-seven dollars and eighty-nine cents"`
    
18. **`million_format_with_commas(amount: float) -> str`**: Includes commas separating million, thousand and hundred groups.
    * Example: `76543210.98` → `"seventy-six million, five hundred forty-three thousand, two hundred ten dollars and ninety-eight cents"`

### Zero Representation Variations

19. **`zero_dollars_explicit_format(amount: float) -> str`**: Explicitly states "zero dollars".
    *   Example: `0.50` → `"zero dollars and fifty cents"`

20. **`no_dollars_format(amount: float) -> str`**: Uses "no dollars" instead of "zero dollars".
    *   Example: `0.50` → `"no dollars and fifty cents"`

21. **`zero_cents_explicit_format(amount: float) -> str`**: Explicitly states "zero cents".
    *   Example: `25.00` → `"twenty-five dollars and zero cents"`

22. **`even_dollars_format(amount: float) -> str`**: Uses "even" for zero cents.
    *   Example: `25.00` → `"twenty-five dollars even"`

23. **`and_no_cents_format(amount: float) -> str`**: Uses "and no cents".
    *   Example: `4325.00` → `"four thousand three hundred twenty-five dollars and no cents"`

### Conjunction Variations

24. **`no_and_format(amount: float) -> str`**: Omits the "and" between dollars and cents.
    *   Example: `25.10` → `"twenty-five dollars ten cents"`

25. **`comma_instead_of_and_format(amount: float) -> str`**: Uses a comma instead of "and".
    *   Example: `25.10` → `"twenty-five dollars, ten cents"`

26. **`ampersand_format(amount: float) -> str`**: Uses an ampersand (&) instead of "and".
    *   Example: `25.10` → `"twenty-five dollars & ten cents"`

27. **`with_preposition_format(amount: float) -> str`**: Uses "with" instead of "and".
    *   Example: `25.10` → `"twenty-five dollars with ten cents"`

28. **`equals_format(amount: float) -> str`**: Uses "equals" to connect dollars and cents.
    *   Example: `5678.90` → `"five thousand six hundred seventy-eight dollars equals ninety cents"`

29. **`totaling_format(amount: float) -> str`**: Uses "totaling" to connect dollars and cents.
    *   Example: `4321.09` → `"four thousand three hundred twenty-one dollars totaling nine cents"`

### Regional and Business-Specific Variations

30. **`formal_business_format(amount: float) -> str`**: A very formal phrasing, starting with "the sum of".
    *   Example: `25478.90` → `"the sum of twenty-five thousand four hundred seventy-eight dollars and ninety cents"`

31. **`legal_document_format(amount: float) -> str`**: Includes the amount in parentheses after the words.
    *   Example: `1250.75` → `"one thousand two hundred fifty dollars and seventy-five cents ($1,250.75)"`

32. **`accounting_format(amount: float) -> str`**:  Uses "USD" and fractional cents.
    *   Example: `83521.46` → `"USD eighty-three thousand five hundred twenty-one and 46/100"`

33. **`corporate_abbreviated_format(amount: float) -> str`**:  Uses abbreviations like "K" and "M".
    *   Example: `12500.00` → `"twelve thousand five hundred dollars (12.5K)"`

34. **`northeast_business_format(amount: float) -> str`**:  A precise Northeastern US style, often ending in "even".
    *   Example: `2568.50` → `"two thousand five hundred sixty-eight dollars and fifty cents even"`

35. **`southern_business_format(amount: float) -> str`**: A Southern US style, often using a comma before the cents.
    *   Example: `4765.21` → `"four thousand seven hundred sixty-five dollars, twenty-one cents"`

36. **`midwest_business_format(amount: float) -> str`**: A direct Midwestern US style.
    *   Example: `3842.50` → `"three thousand eight hundred forty-two dollars fifty cents"`

37. **`west_coast_business_format(amount: float) -> str`**: Includes the numerical amount, a dash, and then the written amount.
    *   Example: `89654.32` → `"$89,654.32 - eighty-nine thousand six hundred fifty-four dollars and thirty-two cents"`

38.  **`fraction_only_centavos_format(amount: float) -> str`**:  Uses "centavos" instead of "cents".
    *   Example: `6789.32` → `"six thousand seven hundred eighty-nine dollars and 32/100 centavos"`

39. **`written_and_numeric_format(amount: float) -> str`**:  Repeats the amount parenthetically in numeric format.
    *   Example: `5432.10` → `"five thousand four hundred thirty-two and 10/100 dollars ($5,432.10)"`

40. **`not_exceeding_format(amount: float) -> str`**:  Uses a "not exceeding" prefix.
    *   Example: `8765.43` → `"not exceeding eight thousand seven hundred sixty-five and 43/100 dollars"`

41. **`business_rounded_format(amount: float) -> str`**:  Highlights when an amount is very close to a round number.
    *   Example: `10000.01` → `"ten thousand dollars and one cent only"`

### Payment and Transaction Variations

42. **`payment_of_format(amount: float) -> str`**:  Begins with "payment of".
    *   Example: `3456.78` → `"payment of three thousand four hundred fifty-six and 78/100 dollars"`

43. **`pay_to_format(amount: float) -> str`**: Uses the phrase "pay to the order of".
    *   Example: `4321.98` → `"pay to the order of four thousand three hundred twenty-one and 98/100 dollars"`

44. **`remittance_format(amount: float) -> str`**: Begins with "remittance of".
    *   Example: `5432.10` → `"remittance of five thousand four hundred thirty-two and 10/100 dollars"`

45. **`funds_transfer_format(amount: float) -> str`**:  Specifies a "transfer of funds".
    *   Example: `9876.54` → `"transfer of funds in the amount of nine thousand eight hundred seventy-six and 54/100 dollars"`

46. **`invoice_payment_format(amount: float) -> str`**:  Specifies an "invoice payment".
    *   Example: `2345.67` → `"invoice payment of two thousand three hundred forty-five and 67/100 dollars"`

47. **`authorized_payment_format(amount: float) -> str`**:  Specifies an "authorized payment".
    *   Example: `8765.43` → `"authorized payment of eight thousand seven hundred sixty-five and 43/100 dollars"`

### Formatting and Punctuation Variations

48. **`extra_commas_format(amount: float) -> str`**: Inserts extra commas within the number words.
    *   Example: `8765.43` → `"eight thousand, seven hundred, sixty-five dollars, and forty-three cents"`

49. **`parenthetical_cents_format(amount: float) -> str`**:  Puts the cents in parentheses.
    *   Example: `9876.54` → `"nine thousand eight hundred seventy-six dollars (and fifty-four cents)"`

50. **`amount_in_words_format(amount: float) -> str`**:  Uses the prefix "amount in words:".
    *   Example: `3456.78` → `"amount in words: three thousand four hundred fifty-six and 78/100 dollars"`

51. **`dollar_amount_of_format(amount: float) -> str`**: Uses the prefix "dollar amount of".
    *   Example: `2345.67` → `"dollar amount of two thousand three hundred forty-five and 67/100"`

52. **`asterisk_bounded_format(amount: float) -> str`**: Surrounds the amount with asterisks.
    *   Example: `5432.10` → `"*five thousand four hundred thirty-two and 10/100 dollars*"`

53. **`duplicate_amount_format(amount: float) -> str`**:  Repeats the entire amount in words.
    *   Example: `3210.98` → `"three thousand two hundred ten and 98/100 dollars (three thousand two hundred ten and 98/100 dollars)"`

### High-Value and Edge Case Variations

54. **`high_value_transaction_format(amount: float) -> str`**:  Precise format for very large amounts.
    *   Example: `25000000.00` → `"twenty-five million dollars and 00/100"`

55. **`multi_million_formal_format(amount: float) -> str`**:  Formal style for multi-million dollar amounts, often ending in "only".
    *   Example: `12345678.90` → `"twelve million three hundred forty-five thousand six hundred seventy-eight dollars and ninety cents only"`

56. **`contract_language_format(amount: float) -> str`**:  Specifies the currency (e.g., "United States dollars") and includes the numeric amount.
    *   Example: `50000000.00` → `"fifty million United States dollars (USD 50,000,000.00)"`

57. **`eight_figure_check_format(amount: float) -> str`**:  Uses "check amount:" prefix for large amounts.
    *   Example: `12345678.90` → `"check amount: twelve million three hundred forty-five thousand six hundred seventy-eight and 90/100 dollars"`

58. **`nine_figure_wire_format(amount: float) -> str`**:  Uses "wire transfer:" prefix for very large amounts.
    *   Example: `123456789.01` → `"wire transfer: one hundred twenty-three million four hundred fifty-six thousand seven hundred eighty-nine and 01/100 dollars"`

59. **`acquisition_payment_format(amount: float) -> str`**: Specifies an "acquisition payment".
    *   Example: `50000000.00` → `"acquisition payment of fifty million dollars and no cents"`

60. **`maximum_liability_format(amount: float) -> str`**:  Uses "maximum liability not to exceed".
    *   Example: `100000000.00` → `"maximum liability not to exceed one hundred million dollars ($100,000,000.00)"`

### Personal Check to Business Variations
61.  **`personal_check_standard_format(amount: float) -> str`**
    -   Standard personal check format
    -   Example: `123.45` → `"one hundred twenty-three and 45/100 dollars"`
    -   Example: `2345.67` → `"two thousand three hundred forty-five and 67/100 dollars"`

62.  **`bill_payment_format(amount: float) -> str`**
    -   Bill payment format (personal to business)
    -   Example: `89.95` → `"eighty-nine dollars and ninety-five cents only"`
    -   Example: `1234.56` → `"one thousand two hundred thirty-four dollars and fifty-six cents only"`

63.  **`donation_check_format(amount: float) -> str`**
    -   Donation check format
    -   Example: `50.00` → `"fifty dollars and 00/100"`
    -   Example: `10000.00` → `"ten thousand dollars and 00/100"`

64.  **`rent_payment_format(amount: float) -> str`**
    -   Rent payment format
    -   Example: `1500.00` → `"one thousand five hundred dollars and 00/100"`
    -   Example: `2750.00` → `"two thousand seven hundred fifty dollars and 00/100"`

65.  **`insurance_payment_format(amount: float) -> str`**
    -   Insurance payment format
    -   Example: `234.56` → `"two hundred thirty-four and 56/100 dollars"`
    -   Example: `1234.56` → `"one thousand two hundred thirty-four and 56/100 dollars"`

66.  **`exact_payment_personal_format(amount: float) -> str`**
    -   Exact amount (personal check emphasis)
    -   Example: `78.99` → `"exact amount of seventy-eight dollars and ninety-nine cents"`
    -   Example: `5678.90` → `"exact amount of five thousand six hundred seventy-eight dollars and ninety cents"`

## Implementation Priority Guidelines

The original priority guidelines are preserved:

1.  **Highest Priority (Business Check Core Formats)**: 1, 2, 3, 6
2.  **High Priority (Common Business Variations)**: 10, 11, 17, 20, 35, 36  (Referencing the *new* numbering)
3.  **Medium Priority (Regional Business Formats)**: 27, 28, 31-34, 53-56 (Referencing the *new* numbering)
4.  **Lower Priority (Personal Check Formats)**: 61-66
5.  **Lowest Priority (Uncommon Variations)**:  The remaining variations.

## TODO

## 3. Synthetic Data Generation (Flat Pattern)

- [ ] Implement MonetaryExpressionGenerator class with flat formatter registry
  - [ ] Create formatter registration system
  - [ ] Implement helper functions for word conversion
  - [ ] Add weighted selection mechanism
  - [ ] Build dataset generation utilities

- [ ] Implement standard format variations (high priority)
  - [ ] Standard dollars and cents format
  - [ ] Fractional cents (X/100) format 
  - [ ] Dollars-after-fraction format
  - [ ] No "and" conjunction format
  - [ ] Only dollars format with "only" suffix
  - [ ] Only cents format

- [ ] Implement number formation variations
  - [ ] Non-hyphenated compound numbers
  - [ ] "And" in numbers over 100
  - [ ] Hundreds as multiples (e.g., "fifteen hundred")
  - [ ] Alternative number forms (dozen, score)

- [ ] Implement zero representation variations
  - [ ] Zero dollars explicit format
  - [ ] "No dollars" format
  - [ ] Zero cents explicit format
  - [ ] "Even" or "flat" for zero cents

- [ ] Implement conjunction variations
  - [ ] Comma instead of "and"
  - [ ] Ampersand format
  - [ ] Preposition format ("with" instead of "and")

- [ ] Implement regional and colloquial variations
  - [ ] "Bits" terminology (e.g., "two bits" for 25¢)
  - [ ] Northeastern variations
  - [ ] Southern variations (singular "dollar" with plural)
  - [ ] Midwestern variations
  - [ ] Pennsylvania Dutch influence formats

- [ ] Implement fractional expressions
  - [ ] Simple fractions ("two and a half dollars")
  - [ ] Quarters format
  - [ ] Mixed notation formats

- [ ] Implement punctuation and formatting variations
  - [ ] Extra commas format
  - [ ] Parenthetical cents format
  - [ ] Slash and dash notations
  - [ ] Plus sign format

- [ ] Implement edge cases and unusual expressions
  - [ ] Very large amounts
  - [ ] Very small amounts
  - [ ] Multiplicative expressions
  - [ ] Mixed denominational terms

- [ ] Add text transformation features
  - [ ] Capitalization variations
  - [ ] Spelling error injection
  - [ ] Spacing issues simulation
  - [ ] Number/word mixing options

- [ ] Create comprehensive test suite
  - [ ] Unit tests for each formatter
  - [ ] Tests for helper functions
  - [ ] Dataset generation tests
  - [ ] Distribution verification tests


## Potential Example Code

import random
import inflect
import json
from typing import Dict, List, Callable, Tuple

# Shared inflect engine
p = inflect.engine()

# Type definition for formatter functions
FormatterFunc = Callable[[float], str]

# Registry to store all formatters with their properties
formatter_registry: Dict[str, Dict] = {}

def register_formatter(name: str, category: str, weight: float = 1.0):
    """Decorator to register a formatter function"""
    def decorator(func: FormatterFunc) -> FormatterFunc:
        formatter_registry[name] = {
            "func": func,
            "category": category,
            "weight": weight,
            "name": name
        }
        return func
    return decorator

# Helper functions
def dollars_to_words(dollars: int) -> str:
    """Convert dollar amount to words"""
    return p.number_to_words(dollars)

def cents_to_words(cents: int) -> str:
    """Convert cents amount to words"""
    return p.number_to_words(cents)

def dollars_to_text(dollars: int) -> str:
    """Format full dollars text with proper plural"""
    dollar_words = dollars_to_words(dollars)
    return f"{dollar_words} {p.plural('dollar', dollars)}"

def cents_to_text(cents: int) -> str:
    """Format cents text with proper plural"""
    cent_words = cents_to_words(cents)
    return f"{cent_words} {p.plural('cent', cents)}"

# Formatter functions
@register_formatter("standard", "standard", weight=10.0)
def standard_format(amount: float) -> str:
    """Standard format: 'Twenty-five dollars and ten cents'"""
    dollars = int(amount)
    cents = int(round((amount - dollars) * 100))
    
    if dollars == 0 and cents == 0:
        return "zero dollars"
    
    result = []
    if dollars > 0:
        result.append(dollars_to_text(dollars))
    
    if cents > 0:
        if dollars > 0:
            result.append("and")
        result.append(cents_to_text(cents))
    
    return " ".join(result)

@register_formatter("fractional_cents", "fractional", weight=5.0)
def fractional_cents_format(amount: float) -> str:
    """Fractional cents: 'Twenty-five dollars and 10/100'"""
    dollars = int(amount)
    cents = int(round((amount - dollars) * 100))
    
    if dollars == 0 and cents == 0:
        return "zero dollars"
    
    result = []
    if dollars > 0:
        result.append(dollars_to_text(dollars))
    
    if dollars > 0 and cents > 0:
        result.append("and")
    
    if cents > 0:
        if dollars == 0:
            result.append(f"{cents}/100 cents")
        else:
            result.append(f"{cents}/100")
    
    return " ".join(result)

@register_formatter("dollars_after_fraction", "fractional", weight=3.0)
def dollars_after_fraction_format(amount: float) -> str:
    """Format: 'Twenty-five and 10/100 dollars'"""
    dollars = int(amount)
    cents = int(round((amount - dollars) * 100))
    
    if dollars == 0 and cents == 0:
        return "zero dollars"
    
    if cents == 0:
        return dollars_to_text(dollars)
    
    if dollars == 0:
        return f"{cents}/100 dollars"
    
    dollar_words = dollars_to_words(dollars)
    return f"{dollar_words} and {cents}/100 dollars"

@register_formatter("no_and", "conjunction", weight=4.0)
def no_and_format(amount: float) -> str:
    """Format without 'and': 'Twenty-five dollars fifty cents'"""
    dollars = int(amount)
    cents = int(round((amount - dollars) * 100))
    
    if dollars == 0 and cents == 0:
        return "zero dollars"
    
    result = []
    if dollars > 0:
        result.append(dollars_to_text(dollars))
    
    if cents > 0:
        result.append(cents_to_text(cents))
    
    return " ".join(result)

@register_formatter("only_dollars", "standard", weight=3.0)
def only_dollars_format(amount: float) -> str:
    """Format for whole dollars: 'Twenty-five dollars only'"""
    dollars = int(amount)
    cents = int(round((amount - dollars) * 100))
    
    if cents > 0:  # This format is only for whole dollar amounts
        return standard_format(amount)
    
    if dollars == 0:
        return "zero dollars"
    
    return f"{dollars_to_text(dollars)} only"

# Add more formatters for all variations...

class MonetaryExpressionGenerator:
    """Generator for monetary expressions"""
    
    def __init__(self):
        self.formatters = formatter_registry
    
    def get_formatters(self, category: str = None) -> List[Dict]:
        """Get all formatters, optionally filtered by category"""
        if category:
            return [fmt for fmt in self.formatters.values() if fmt["category"] == category]
        return list(self.formatters.values())
    
    def get_categories(self) -> List[str]:
        """Get all available categories"""
        return list(set(fmt["category"] for fmt in self.formatters.values()))
    
    def generate(self, amount: float, formatter_name: str = None) -> Tuple[str, str]:
        """Generate expression using specific formatter or random with weights"""
        if formatter_name and formatter_name in self.formatters:
            formatter = self.formatters[formatter_name]
            expression = formatter["func"](amount)
            return expression, formatter["name"]
        
        # Select random formatter with weights
        formatters = list(self.formatters.values())
        weights = [fmt["weight"] for fmt in formatters]
        formatter = random.choices(formatters, weights=weights, k=1)[0]
        
        expression = formatter["func"](amount)
        return expression, formatter["name"]
    
    def apply_text_transformations(self, text: str, typo_prob: float = 0.02,
                                  cap_prob: float = 0.05, spacing_prob: float = 0.03) -> str:
        """Apply random transformations like capitalization, spacing, typos"""
        # Implementation of text transformations...
        return text
        
    def generate_dataset(self, amounts: List[float], output_path: str,
                         apply_transformations: bool = True):
        """Generate dataset from a list of amounts"""
        examples = []
        
        for amount in amounts:
            expression, formatter_name = self.generate(amount)
            
            if apply_transformations:
                expression = self.apply_text_transformations(expression)
                
            # Create example with metadata
            examples.append({
                "input": expression,
                "output": json.dumps({"amount": round(amount, 2)}),
                "amount": round(amount, 2),
                "formatter": formatter_name
            })
            
        # Write to output file
        with open(output_path, 'w') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')