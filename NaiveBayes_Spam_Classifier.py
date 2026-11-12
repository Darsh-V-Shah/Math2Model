import math
from collections import Counter


# 70 spam + 100 ham, pre-labeled.
emails = [
    # spam
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

    # ham
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


# ------------------------------------------------------------
# Naive Bayes helpers
# ------------------------------------------------------------

def tokenize(email):
    return email.lower().split()


def build_vocabulary(emails):
    vocab = set()
    for text, _ in emails:
        vocab.update(tokenize(text))
    return vocab


def count_words_by_class(emails):
    spam_counts = Counter()
    ham_counts = Counter()
    spam_total = 0
    ham_total = 0

    for text, label in emails:
        for word in tokenize(text):
            if label == "Spam":
                spam_counts[word] += 1
                spam_total += 1
            elif label == "Ham":
                ham_counts[word] += 1
                ham_total += 1

    return spam_counts, ham_counts, spam_total, ham_total


def calculate_priors(emails):
    total = len(emails)
    spam = sum(1 for _, label in emails if label == "Spam")
    ham = sum(1 for _, label in emails if label == "Ham")
    return spam / total, ham / total


def calculate_word_probabilities(word_counts, total_words, vocab_size, alpha=1):
    denom = total_words + alpha * vocab_size
    return {word: (count + alpha) / denom for word, count in word_counts.items()}


def get_word_probability(word, word_counts, total_words, vocab_size, alpha=1):
    # Laplace smoothing: unseen words still get a small non-zero chance.
    count = word_counts.get(word, 0)
    return (count + alpha) / (total_words + alpha * vocab_size)


def predict(email, spam_prior, ham_prior, spam_counts, ham_counts,
            spam_total, ham_total, vocab_size, alpha=1):
    # Compare log P(Spam) + sum log P(w|Spam) against the Ham equivalent.
    # Logs because multiplying dozens of sub-1 probabilities underflows fast.
    words = tokenize(email)

    spam_log = math.log(spam_prior)
    ham_log = math.log(ham_prior)
    for word in words:
        spam_log += math.log(get_word_probability(word, spam_counts, spam_total, vocab_size, alpha))
        ham_log += math.log(get_word_probability(word, ham_counts, ham_total, vocab_size, alpha))

    return ("Spam" if spam_log > ham_log else "Ham"), spam_log, ham_log


# ------------------------------------------------------------
# Run it
# ------------------------------------------------------------

vocabulary = build_vocabulary(emails)
print("Vocabulary size =", len(vocabulary))

spam_word_counts, ham_word_counts, spam_total_words, ham_total_words = \
    count_words_by_class(emails)

print("\nSpam word counts:", spam_word_counts)
print("\nHam word counts:", ham_word_counts)
print("\nTotal Spam words =", spam_total_words)
print("Total Ham words =", ham_total_words)

spam_prior, ham_prior = calculate_priors(emails)
print("\nP(Spam) =", spam_prior)
print("P(Ham)  =", ham_prior)

# Computed for completeness; predict() looks words up on the fly.
spam_probabilities = calculate_word_probabilities(
    spam_word_counts, spam_total_words, len(vocabulary))
ham_probabilities = calculate_word_probabilities(
    ham_word_counts, ham_total_words, len(vocabulary))

test_email = "See you at the meeting tomorrow."

prediction, spam_score, ham_score = predict(
    test_email,
    spam_prior, ham_prior,
    spam_word_counts, ham_word_counts,
    spam_total_words, ham_total_words,
    len(vocabulary),
)

print("\nTest email:", test_email)
print("Spam log score =", spam_score)
print("Ham log score  =", ham_score)
print("Prediction     =", prediction)
