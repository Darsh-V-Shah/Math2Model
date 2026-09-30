import math
from collections import Counter


# ============================================================
# 1. TRAINING DATA
# ============================================================

emails = [

    # =========================
    # SPAM EMAILS - 70
    # =========================

    ("free money", "Spam"),
    ("win prize", "Spam"),
    ("free prize", "Spam"),
    ("claim your reward", "Spam"),
    ("you won cash", "Spam"),
    ("urgent offer", "Spam"),
    ("limited time offer", "Spam"),
    ("click to win", "Spam"),
    ("free gift waiting", "Spam"),
    ("congratulations winner", "Spam"),
    ("earn money fast", "Spam"),
    ("make money online", "Spam"),
    ("exclusive deal", "Spam"),
    ("cash bonus", "Spam"),
    ("get free cash", "Spam"),
    ("win a lottery", "Spam"),
    ("lottery winner", "Spam"),
    ("claim free voucher", "Spam"),
    ("special promotion", "Spam"),
    ("act now", "Spam"),
    ("urgent claim", "Spam"),
    ("you are selected", "Spam"),
    ("free vacation", "Spam"),
    ("win a trip", "Spam"),
    ("bonus waiting", "Spam"),
    ("cheap loans", "Spam"),
    ("easy money", "Spam"),
    ("work from home", "Spam"),
    ("earn extra cash", "Spam"),
    ("double your income", "Spam"),
    ("free coupon", "Spam"),
    ("claim your coupon", "Spam"),
    ("discount prize", "Spam"),
    ("you have won", "Spam"),
    ("cash prize available", "Spam"),
    ("free tickets", "Spam"),
    ("buy now save money", "Spam"),
    ("hot deal today", "Spam"),
    ("exclusive prize", "Spam"),
    ("limited offer", "Spam"),
    ("get rich quickly", "Spam"),
    ("investment opportunity", "Spam"),
    ("guaranteed profit", "Spam"),
    ("cheap price", "Spam"),
    ("free membership", "Spam"),
    ("click this link", "Spam"),
    ("claim reward now", "Spam"),
    ("special bonus", "Spam"),
    ("winner notification", "Spam"),
    ("congratulations you won", "Spam"),
    ("free sample", "Spam"),
    ("discount offer", "Spam"),
    ("lowest price", "Spam"),
    ("urgent response needed", "Spam"),
    ("claim your bonus", "Spam"),
    ("instant cash", "Spam"),
    ("earn thousands", "Spam"),
    ("free entry", "Spam"),
    ("win big", "Spam"),
    ("prize waiting", "Spam"),
    ("money transfer offer", "Spam"),
    ("special lottery", "Spam"),
    ("free trial offer", "Spam"),
    ("get your reward", "Spam"),
    ("act immediately", "Spam"),
    ("exclusive bonus", "Spam"),
    ("cash available", "Spam"),
    ("limited discount", "Spam"),
    ("free shopping voucher", "Spam"),
    ("you are a winner", "Spam"),


    # =========================
    # HAM EMAILS - 100
    # =========================

    ("meeting tomorrow", "Ham"),
    ("project meeting", "Ham"),
    ("see you tomorrow", "Ham"),
    ("team meeting at noon", "Ham"),
    ("please send the report", "Ham"),
    ("lunch at two", "Ham"),
    ("can we meet today", "Ham"),
    ("class starts at nine", "Ham"),
    ("assignment is due Friday", "Ham"),
    ("please review this document", "Ham"),
    ("see you in class", "Ham"),
    ("call me after lunch", "Ham"),
    ("the meeting was postponed", "Ham"),
    ("project deadline is Monday", "Ham"),
    ("send me the notes", "Ham"),
    ("thanks for your help", "Ham"),
    ("where are you", "Ham"),
    ("are you coming today", "Ham"),
    ("lets meet after class", "Ham"),
    ("please check the file", "Ham"),
    ("the report looks good", "Ham"),
    ("can you call me", "Ham"),
    ("I will be late", "Ham"),
    ("see you soon", "Ham"),
    ("what time is the meeting", "Ham"),
    ("please bring your laptop", "Ham"),
    ("the class was cancelled", "Ham"),
    ("submit the assignment today", "Ham"),
    ("thank you for the update", "Ham"),
    ("I sent the document", "Ham"),
    ("please reply when free", "Ham"),
    ("see you tomorrow morning", "Ham"),
    ("can we discuss the project", "Ham"),
    ("the presentation is ready", "Ham"),
    ("I need your feedback", "Ham"),
    ("lets work together", "Ham"),
    ("please share the notes", "Ham"),
    ("the exam is next week", "Ham"),
    ("study group tonight", "Ham"),
    ("meet me at the library", "Ham"),
    ("I finished the work", "Ham"),
    ("the file is attached", "Ham"),
    ("please check your email", "Ham"),
    ("thanks for sending this", "Ham"),
    ("I will send it later", "Ham"),
    ("what is the homework", "Ham"),
    ("the lecture starts soon", "Ham"),
    ("can you help me", "Ham"),
    ("I have a question", "Ham"),
    ("lets talk after class", "Ham"),
    ("the room has changed", "Ham"),
    ("please arrive on time", "Ham"),
    ("the project is going well", "Ham"),
    ("I will join the meeting", "Ham"),
    ("thanks for the information", "Ham"),
    ("see you at lunch", "Ham"),
    ("please update the document", "Ham"),
    ("the report is attached", "Ham"),
    ("I completed the assignment", "Ham"),
    ("can we reschedule", "Ham"),
    ("the meeting starts at ten", "Ham"),
    ("I will call tomorrow", "Ham"),
    ("please remind me", "Ham"),
    ("I am working on it", "Ham"),
    ("the presentation went well", "Ham"),
    ("see you next week", "Ham"),
    ("can you send the link", "Ham"),
    ("please check the schedule", "Ham"),
    ("I will be there", "Ham"),
    ("thanks for your message", "Ham"),
    ("lets meet tomorrow", "Ham"),
    ("the deadline is approaching", "Ham"),
    ("please send the details", "Ham"),
    ("I received your email", "Ham"),
    ("the class starts soon", "Ham"),
    ("can you review this", "Ham"),
    ("I will finish today", "Ham"),
    ("see you at the meeting", "Ham"),
    ("please bring the report", "Ham"),
    ("the project meeting is today", "Ham"),
    ("I need the file", "Ham"),
    ("thanks for checking", "Ham"),
    ("the assignment is ready", "Ham"),
    ("I will send the notes", "Ham"),
    ("can we meet tomorrow", "Ham"),
    ("please let me know", "Ham"),
    ("I am available today", "Ham"),
    ("the meeting room is booked", "Ham"),
    ("I finished the report", "Ham"),
    ("see you after class", "Ham"),
    ("please share the file", "Ham"),
    ("the exam schedule is out", "Ham"),
    ("can you send me the document", "Ham"),
    ("I will arrive soon", "Ham"),
    ("thanks for your reply", "Ham"),
    ("lets discuss this tomorrow", "Ham"),
    ("the work is almost done", "Ham"),
    ("please attend the meeting", "Ham"),
    ("I will bring the laptop", "Ham"),
    ("the project is complete", "Ham"),
    ("see you in the morning", "Ham"),
    ("please send the assignment", "Ham"),
]


# ============================================================
# 2. PREPROCESSING
# ============================================================

def tokenize(email):
    """
    Convert an email into a list of words.

    Example:
        "Free Money" -> ["free", "money"]
    """

    email = email.lower()

    # Simple tokenization:
    words = email.split()

    return words


# ============================================================
# 3. BUILD THE VOCABULARY
# ============================================================

def build_vocabulary(emails):
    """
    Create a set containing every unique word
    appearing in the training emails.
    """

    vocabulary = set()

    for email, label in emails:

        words = tokenize(email)

        for word in words:
            vocabulary.add(word)

    return vocabulary


# ============================================================
# 4. COUNT WORDS FOR EACH CLASS
# ============================================================

def count_words_by_class(emails):
    """
    Count how many times each word appears in Spam
    and Ham emails.
    """

    spam_word_counts = Counter()
    ham_word_counts = Counter()

    spam_total_words = 0
    ham_total_words = 0

    for email, label in emails:

        words = tokenize(email)

        if label == "Spam":

            for word in words:
                spam_word_counts[word] += 1
                spam_total_words += 1

        elif label == "Ham":

            for word in words:
                ham_word_counts[word] += 1
                ham_total_words += 1

    return (
        spam_word_counts,
        ham_word_counts,
        spam_total_words,
        ham_total_words
    )


# ============================================================
# 5. CALCULATE CLASS PRIORS
# ============================================================

def calculate_priors(emails):
    """
    Calculate:

        P(Spam)
        P(Ham)
    """

    total_emails = len(emails)

    spam_emails = 0
    ham_emails = 0

    for email, label in emails:

        if label == "Spam":
            spam_emails += 1

        elif label == "Ham":
            ham_emails += 1

    spam_prior = spam_emails / total_emails
    ham_prior = ham_emails / total_emails

    return spam_prior, ham_prior


# ============================================================
# 6. CALCULATE WORD PROBABILITIES
# ============================================================

def calculate_word_probabilities(
        word_counts,
        total_words,
        vocabulary_size,
        alpha=1
):
    """
    Calculate:

        P(word | class)

    using Laplace smoothing.

    Formula:

        P(word | class) =
        (count(word, class) + alpha)
        --------------------------------
        total_words + alpha * V

    where:

        V = vocabulary size
        alpha = smoothing parameter
    """

    probabilities = {}

    denominator = total_words + alpha * vocabulary_size

    for word, count in word_counts.items():

        probability = (count + alpha) / denominator

        probabilities[word] = probability

    return probabilities


# ============================================================
# 7. GET PROBABILITY FOR A WORD
# ============================================================

def get_word_probability(
        word,
        word_counts,
        total_words,
        vocabulary_size,
        alpha=1
):
    """
    Return P(word | class).

    Notice that we also need to handle words that
    never appeared in that class.

    Example:

        P(meeting | Spam)

    meeting appeared 0 times in Spam.

    Laplace smoothing gives:

        (0 + 1) / (total_words + V)
    """

    count = word_counts.get(word, 0)

    denominator = total_words + alpha * vocabulary_size

    probability = (count + alpha) / denominator

    return probability


# ============================================================
# 8. PREDICT EMAIL
# ============================================================

def predict(
        email,
        spam_prior,
        ham_prior,
        spam_word_counts,
        ham_word_counts,
        spam_total_words,
        ham_total_words,
        vocabulary_size,
        alpha=1
):
    """
    Predict whether an email is Spam or Ham.

    We use log probabilities:

        log P(Class)
        +
        sum(log P(word | Class))

    This prevents numerical underflow when emails
    contain many words.
    """

    words = tokenize(email)

    # --------------------------------------------------------
    # Spam score
    # --------------------------------------------------------

    spam_log_score = math.log(spam_prior)

    for word in words:

        probability = get_word_probability(
            word,
            spam_word_counts,
            spam_total_words,
            vocabulary_size,
            alpha
        )

        spam_log_score += math.log(probability)

    # --------------------------------------------------------
    # Ham score
    # --------------------------------------------------------

    ham_log_score = math.log(ham_prior)

    for word in words:

        probability = get_word_probability(
            word,
            ham_word_counts,
            ham_total_words,
            vocabulary_size,
            alpha
        )

        ham_log_score += math.log(probability)

    # --------------------------------------------------------
    # Compare scores
    # --------------------------------------------------------

    if spam_log_score > ham_log_score:
        prediction = "Spam"

    else:
        prediction = "Ham"

    return prediction, spam_log_score, ham_log_score


# ============================================================
# 9. MAIN PROGRAM
# ============================================================

# ------------------------------------------------------------
# Build vocabulary
# ------------------------------------------------------------

vocabulary = build_vocabulary(emails)

"""print("=" * 60)
print("VOCABULARY")
print("=" * 60)

print(vocabulary)
print()"""

print("Vocabulary size =", len(vocabulary))


# ------------------------------------------------------------
# Count words
# ------------------------------------------------------------

(
    spam_word_counts,
    ham_word_counts,
    spam_total_words,
    ham_total_words
) = count_words_by_class(emails)


print()
print("=" * 60)
print("WORD COUNTS")
print("=" * 60)

print("\nSpam word counts:")
print(spam_word_counts)

print("\nHam word counts:")
print(ham_word_counts)

print("\nTotal Spam words =", spam_total_words)
print("Total Ham words =", ham_total_words)


# ------------------------------------------------------------
# Calculate priors
# ------------------------------------------------------------

spam_prior, ham_prior = calculate_priors(emails)

print()
print("=" * 60)
print("CLASS PRIORS")
print("=" * 60)

print("P(Spam) =", spam_prior)
print("P(Ham)   =", ham_prior)


# ------------------------------------------------------------
# Calculate word probabilities
# ------------------------------------------------------------

spam_probabilities = calculate_word_probabilities(
    spam_word_counts,
    spam_total_words,
    len(vocabulary)
)

ham_probabilities = calculate_word_probabilities(
    ham_word_counts,
    ham_total_words,
    len(vocabulary)
)


"""print()
print("=" * 60)
print("WORD PROBABILITIES")
print("=" * 60)

print("\nSpam probabilities:")

for word in sorted(vocabulary):

    probability = get_word_probability(
        word,
        spam_word_counts,
        spam_total_words,
        len(vocabulary)
    )

    print(f"P({word} | Spam) = {probability:.4f}")


print("\nHam probabilities:")

for word in sorted(vocabulary):

    probability = get_word_probability(
        word,
        ham_word_counts,
        ham_total_words,
        len(vocabulary)
    )

    print(f"P({word} | Ham) = {probability:.4f}")"""


# ============================================================
# 10. TEST EMAIL
# ============================================================

test_email = "See you at the meeting tomorrow."

print()
print("=" * 60)
print("PREDICTION")
print("=" * 60)

print("Test email:", test_email)


prediction, spam_score, ham_score = predict(
    test_email,
    spam_prior,
    ham_prior,
    spam_word_counts,
    ham_word_counts,
    spam_total_words,
    ham_total_words,
    len(vocabulary)
)


print()
print("Spam log score =", spam_score)
print("Ham log score  =", ham_score)

print()
print("Prediction =", prediction)
