

############################################################
# Imports
############################################################

import math
import email
import os
import time


############################################################
# Section 1: Spam Filter
############################################################

def load_tokens(email_path):
    answer=[]
    file_obj = open(email_path,encoding="utf8")
    message = email.message_from_file(file_obj)
    msg_iterator = email.iterators.body_line_iterator(message)
    file_obj.close()
    for line in msg_iterator:
        for word in line.split():
            answer.append(word)
    return answer
'''
ham_dir="data/train/ham/"
print(load_tokens(ham_dir+"ham1")[200:204])
print(load_tokens(ham_dir+"ham2")[110:114])
spam_dir="data/train/spam/"
print(load_tokens(spam_dir+"spam1")[1:5])
print(load_tokens(spam_dir+"spam2")[:4])
print("\n")'''

def log_probs(email_paths, smoothing):
    tokens=[]
    word_dict={}
    log_dict={}
    for path in email_paths:
        for answer in load_tokens(path):
            tokens.append(answer)
    tot_words = len(tokens)
    
    for w in tokens:
        if w in word_dict:
            word_dict[w]+=1
        else:
            word_dict[w]=1
    V=len(word_dict)
    denominator = (tot_words + (smoothing * (V + 1)))
    for word in word_dict:
        log_dict[word]=math.log((word_dict[word] + smoothing) / denominator)
    log_dict["<UNK>"] = math.log(smoothing / denominator)
    return log_dict
'''
paths=["data/train/ham/ham%d"%i for i in range(1,11)]
p=log_probs(paths,1e-5)
print(p["the"])
print(p["line"])
paths=["data/train/spam/spam%d"%i for i in range(1,11)]
p=log_probs(paths,1e-5)
print(p["Credit"])
print(p["<UNK>"])
print("\n")'''

class SpamFilter(object):

    def __init__(self, spam_dir, ham_dir, smoothing):
        spam = [spam_dir + "/" + s for s in os.listdir(spam_dir)]
        ham = [ham_dir + "/" + s for s in os.listdir(ham_dir)]
        
        self.spam_dict = log_probs(spam, smoothing)
        self.ham_dict = log_probs(ham, smoothing)
        
        self.prob_spam = math.log(len(os.listdir(spam_dir)) / float(len(os.listdir(spam_dir)) + len(os.listdir(ham_dir))))
        self.prob_ham = math.log(1 - self.prob_spam)
    
    def is_spam(self, email_path):
        tokens=[]
        word_dict={}
        ans_spam=0
        ans_ham=0
        prob_spam=self.prob_spam
        prob_ham=self.prob_ham
        for answer in load_tokens(email_path):
            tokens.append(answer)
        tot_words = len(tokens)
        
        for w in tokens:
            if w in word_dict:
                word_dict[w]+=1
            else:
                word_dict[w]=1        

        for word in word_dict.keys():
            if word in self.spam_dict:
                ans_spam+=self.spam_dict[word]*word_dict[word]
            else:
                ans_spam+=self.spam_dict["<UNK>"]*word_dict[word]
                
            if word in self.ham_dict:
                ans_ham+=self.ham_dict[word]*word_dict[word]
            else:
                ans_ham+=self.ham_dict["<UNK>"]*word_dict[word]
                
        if ans_spam > ans_ham:
            return True
        else:
            return False
        
       
    def most_indicative_spam(self, n):
        ans_dict = {}
        prob_spam = math.exp(self.prob_spam)
        prob_ham = math.exp(self.prob_ham)
        for word in (self.spam_dict):
            if word in self.ham_dict:
                numerator = math.exp(self.spam_dict[word])
                denominator1 = float(math.exp(self.spam_dict[word]) * prob_spam)
                denominator2 = float(math.exp(self.ham_dict[word]) * prob_ham)
                ans_dict[word] = math.log(numerator / (denominator1 + denominator2))
        answer=list(sorted(ans_dict, key=ans_dict.get, reverse=True))
        return answer[:n]
       
    def most_indicative_ham(self, n):
        ans_dict = {}
        prob_spam = math.exp(self.prob_spam)
        prob_ham = math.exp(self.prob_ham)
        for word in (self.spam_dict):
            if word in self.ham_dict:
                numerator = math.exp(self.ham_dict[word])
                denominator1 = float(math.exp(self.spam_dict[word]) * prob_spam)
                denominator2 = float(math.exp(self.ham_dict[word]) * prob_ham)
                ans_dict[word] = math.log(numerator / (denominator1 + denominator2))
        answer=list(sorted(ans_dict, key=ans_dict.get, reverse=True))
        return answer[:n]
'''
sf=SpamFilter("data/train/spam","data/train/ham",1e-5)
print(sf.is_spam("data/train/spam/spam1"))
print(sf.is_spam("data/train/spam/spam2"))
sf=SpamFilter("data/train/spam","data/train/ham",1e-5)
print(sf.is_spam("data/train/ham/ham1"))
print(sf.is_spam("data/train/ham/ham2"))
print("\n")

sf=SpamFilter("data/train/spam","data/train/ham",1e-5)
print(sf.most_indicative_spam(5))
sf=SpamFilter("data/train/spam","data/train/ham",1e-5)
print(sf.most_indicative_ham(5))'''

