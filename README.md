# Spam-detection-FYP
                                                     EMAIL SPAM DETECTION
Email Spam Detection, including machine learning
Abstract
Since the commercialization of the internet, there has been immense growth in the number of people using the internet, and, as such, the popularity of email in society has risen, whether personal or business-oriented. Equally, there has been a great rise in the amount of spam emails plaguing consumers' inboxes. Spam emails can be defined as unsolicited messages sent in bulk by email. These can be phishing emails created by cybercriminals that prey on users, tricking them into divulging personal security information, downloading malicious malware onto their devices, or scamming them into giving them money. This project aims to develop a clear method for distinguishing between “Spam” (false) and “Ham” (legitimate) emails, enabling users to access their emails safely and efficiently without risk of losing sensitive information. It will include a detailed analysis of the types of spam that have continued to plague us in modern society, as well as the methods used to filter emails. It will also discuss the spam filtering methods used by mainstream email platforms such as Gmail and Yahoo. Finally, it will include a machine learning model that uses multiple  Natural Language Processing (NLP) techniques that are used to detect spam emails.
Keywords: Spam detection, Email Filtering, Phishing, Machine learning, Naïve Bayes, NLP, 
Spambots.

1)	Introduction
The world has become increasingly reliant on technology over time, with individual consumers and businesses utilizing it to handle the majority of their financial transactions and other business engagements. With the heavy reliance on technology came an increase in the number of cybercriminals and online phishing scams, one of which is spam emails. Spam emails are unsolicited mail sent in bulk to users through email services by botnets or cybercriminals. The cybercriminals can use these emails to gain access to the user's sensitive files and information by tricking the user into entering the information or forcefully downloading dangerous malware onto the user's device through “Spam links”. These attacks don’t just impact emails; they can also be sent through text or social media platforms, increasing the difficulty in detecting and avoiding them for the average user.
Email users waste a significant amount of time and effort sorting through their emails trying to get rid of the spam in their inboxes. This would not only affect the average person but also employees of major companies. It would affect the efficiency of employees in completing tasks, as emails are used heavily for chain communication as well as to delegate work. Having to deal with spam while working can cause the users themselves irritation and stress from filtering their inboxes from hundreds of spam emails. As a result, an effective spam email filter is paramount for the efficiency of users traversing their emails. It would relieve the burden users have by filtering out all the harmful emails, allowing them to effectively communicate with employees or friends without the fear of losing private information or downloading malware. 


The aims of this paper will be:
•	Understanding the aim of spam.
•	Analyse existing spam email filters used by major companies
•	Examine spam email detection and its techniques.
•	Provide a spam email detection model.

2)	Phishing scams
2.1) Phishing
Phishing represents one of the most persuasive and damaging threats users face in the digital age. Phishing is defined as the act of cybercriminals posing as a legitimate business, institution, or individual to deceive people into revealing sensitive, confidential information, such as usernames, passwords, or social security data. The consequences linked with phishing can be devastating to an individual, leading to identity theft, financial loss, or reputational damage[3]. As technology has advanced, it is crucial for consumers to stay updated on not only how these attacks are conducted but also how they have evolved over time. Phishing has been prevalent since the internet was released to the general public, with the average person and high-profile businesses using their personal/confidential information on a widely accessible platform, creating a breeding ground for cybercriminals to infiltrate and attempt to steal data. Spam is a common phishing technique used to enact phishing attacks on users. Some of these types include:
2.2) Email Spam
Email spam, or as it's widely known as “junk mail,” is the most prevalent type of spam users face today. It is unsolicited emails sent in bulk to a large number of recipients via email. These emails will not only clutter up users' inboxes but also pose a security risk if the user clicks on any malicious links or downloads incorporated in the emails. On a larger scale, email spam would pose a significant threat to any business. The constant bombardment of spam clutters employees' inboxes, leading to strain on the company's servers and networks, increasing the likelihood of the servers crashing or slowing down, as well as hindering the workers' efficiency in completing tasks. There is also a security risk for companies, as these scams can lead to cybercriminals obtaining sensitive information, causing security breaches, which would cause the company to waste time and effort to rectify.
2.3) Email Spoofing
Email spoofing is the deceptive practice in which the sender appropriates the identity of a trusted source. This is one of the most effective phishing techniques cybercriminals use because, by falsifying the sender's identity, they can exploit the user’s trust, which will increase the chances of the user opening their email and carrying out any action they suggest.[3]
The first substantial phishing attack was a spoofing incident that occurred in the mid-1990s when cybercriminals would pose as America Online (AOL) employees and use instant messaging and spoofed emails to steal users’ login credentials(Ever Nimble 2024). This was a huge incident at the time, as AOL was one of the leading internet service providers for consumers. The cybercriminals would not only pose as AOL employees to gain information, but also as Famous or rich individuals. The cybercriminal would promise the victim large sums of money in exchange for payments made upfront or for providing them with personal information needed to claim it. 
2.4) Social media spam
Social media spam refers to the repetitive, unsolicited, and deceptive content posted online. These include the promotion of bad products, the spread of misinformation, or malware links[14]. Unlike email spam, this form of spam can be more difficult to detect, as it is more interactive. Spammers may use bots to flood the timeline, group chats, or posts to make their scams look more legitimate. Some cybercriminals use it to boost engagement inorganically or scam users into buying a faulty or fake product. Social media spam can be tied heavily to trends, making it harder to spot.

2.5) Fake news/Current event spam
Fake news or current event scams are schemes that exploit major news, current topics, and crises to deceive people and steal money[14]. The spam of this sort can be seen on social media platforms, websites, or emails. They can appear in the form of pop-up ads, advertisements, and false emails. Cybercriminals take advantage of the users’ heightened emotions over natural disasters, politics, or large-scale current events to craft convincing scams that are timely and appear to be legitimate. These scams can promise exclusive information, urgent relief efforts for those in need, and special offers tied to events; however, all are fraudulent.
2.6) Malware advertising spam
Malware advertising spam, also known as “malvertising”, is a cyberattack method where malicious software is distributed through online advertisements. Unlike traditional email spam, malvertising can appear on legitimate websites through fraudulent or compromised ad networks, making it more difficult for users to differentiate from legitimate ads. These ads will promise users a type of reward for interacting with the advertisement, such as free software or prizes, luring the user to click[14]. Once clicked, the user would be redirected to infected websites, which would automatically download infected software onto the computer, such as spyware or trojans, which steal sensitive data, monitor the users’ browsing habits, or hijack the system's resources[3]. This method is highly effective as online advertising networks distribute ads across countless websites; in turn, a single malicious campaign ad can reach millions of users in a short time.
2.7) Spambots
Previously, before the early 2000’s cybercriminals mostly conducted individually and were carried out completely by human operators[16]. However, with the introduction of spambots, cybercriminals are able to carry out their schemes with less effort. Spambots are automated robots designed to distribute spam schemes with minimal human intervention.
There is no limit to the number of spambots a cybercriminal can use, meaning they can run multiple mass phishing attacks on users with next to no effort. Cybercriminals are also able to train the spam bots to adapt to use trends and habits, making them benign. As a result, the schemes are easier for them to conduct.
An example of this would be a spam bot type called “Social bot,” which is a bot that is set up with a computer algorithm that can emulate the actions of real people while being able to adapt to different situations, altering their behaviour[16]. With the incorporation of spam bots online, the act of phishing has become less tedious and expensive for cybercriminals, in turn scaling the scam operations to an unprecedented level. The use of Spambots increases the difficulty in fighting spam immensely, as they can carry out the schemes without human error with higher efficiency.

3)	Email spam filtering
As cybercriminals were running rampant, the technology industry had to adapt quickly to the rise in phishing scandals. With the increasing rise of phishing scandals plaguing the internet, countermeasures needed to be created to counteract phishing and defend users. 
3.1) Heuristic filtering
One of the first forms of anti-phishing software was heuristic filtering, which was introduced in the mid-1990s. The heuristic filters would work by assigning a score to different elements of an email. The score would rely on the presence of certain keywords, the use of capitalisation, and other spam keywords[15]. The goal of the filters was to sift through emails and reduce the false positives in the email structure, making it less likely to be flagged as spam or a scam. This was the development of the first substantial anti-phishing software. Unfortunately, with the rapid development of the internet, cybercriminals were able to adapt to the implementation of anti-phishing software and find heuristic evasion techniques to get around the system. This would spark the phishing wars, where companies would develop their software to counteract cybercriminals, while the latter would find methods to circumvent the systems and continue phishing consumers[13]
3.2) Gmail Spam filtering:
Gmail is the most popular Gmail service in modern times, with a staggering 1.8 billion active users and 121 billion emails sent daily. As a result, they have to be proactive when it comes to the filtering of spam emails to ensure a safe experience for their users. Gmail operates with its in-house filter system called “Gmail Spam filter”[11]. This system funnels emails through a sort of checkpoint queue evaluation, using key factors to determine if the email is spam or legitimate. These factors include the email's origin(IP address of the sender), domain name, and whether users have previously flagged a certain email as spam; their systems can proactively adapt to combat cybercriminals by utilizing information from their users[12]. By using these filtering techniques, Gmail was able to drop the number of unauthenticated emails by 75%[11].

3.3) Outlook Spam filtering:
Outlook incorporates its own built-in spam filtering system called “Junk Email Filter”[10]. They integrate an upgraded and modernised iteration of Heuristic filtering, which relies on certain phrases, unusual sender information, suspicious attachments, and inconsistent patterns in combination to deduce whether an email is spam or not. They also complement this system with user feedback similar to Google to combat the evolving nature of spam.
Another technique they use in combination is “Bayesian filtering”. This technique concerns machine learning using statistical methods instead of focusing on the literal side of the spam emails[10]. It learns from the previous filtering content that has been flagged by the system and gathers data over time to become more accurate in detecting spam. It contains data annotation in order to improve the algorithm and improve detection. Since cybercriminals are adapting to the new filtering systems, major email services have to continuously update their filtering systems. These systems are effective because they can detect new tactics cybercriminals may use and even pick up subtle nuances, as a result, improving the accuracy of the filter system.
When it comes to implementing an effective spam filter model, the key element is the accumulation of user data that Gmail and Outlook gather. With how hard it is to constantly spot phishing attempts since they are continuously adapting to current affairs and improving on their deception, this allows them to spot the cybercriminals' schemes before they can reach a substantial number of users.
4)	Literature Review
4.1) Dataset
There has been a huge advancement in the software used to detect spam since the use of basic heuristic filtering. Major email companies are continually improving their systems to enhance their efficiency. My Email Spam Model uses Natural Language Processing(NLP) to separate ham and spam using machine learning. To start, my dataset’s name is “Email Spam Detection,” which I obtained from Kaggle.com[1]. It contains 5572 entries, split between spam and ham, which are given to each algorithm as an input. 5572 of the texts are unique, so the model can be tested on its ability to detect new and existing spam that’s been seen previously. I chose this data as it has a large range of data, so the models can have distinct differences in results. The dataset includes two columns named “message”(holding all the sample text) and “Category” holding the values of the text). The Category column consists of numbers indicating if the sample text is spam(1) or ham(0).









Fig1.  Representation of an equal number of spam and ham for accurate results















          Fig 2. Visual Representation of text length in the emails, X-axis showing the text length, and the Y-axis showing the Text length.

4.2) Performance metrics
There will be three algorithms tested within the model, and the performance will be documented and compared through True positive(TP), True Negative(TN), False positive(FP), and False negative(FN). These metrics provide a detailed insight into the detection accuracy[6]. These elements will combine to give us the accuracy, precision, recall, and F1 score. On top of that, I have included the CPU time and Wall time to gauge how these algorithms affect the systems used. 
True Positive(TP) refers to the number of instances that the model had correctly predicted in the positive class. True Negative(TN) indicates the number of instances where the model correctly predicted the negative class. False Positive(FP) is the number of instances where the model incorrectly predicted the positive class. Finally, False negatives (FN) are the number of instances where the model incorrectly predicted the negative class.[2]
1)Accuracy represents the number of correctly classified data instances over the total number of data instances. It's calculated, though this calculation:[5]



2)Precision refers to the number of correct calculations that the model returns. It should be as close to 1 as possible. It is calculated through the following formula:[5]

3)Recall should also ideally be as close to 1 as possible it considers all the actual positives, not all the correct classifications. It is calculated through this formula:[5]

4)F1 Score is the mean of precision and recall, giving equal weight to both, which proves the balanced measure of the model's accuracy. It can be represented by this formula:[5]

5)CPU time refers to the amount of time a computer's central processing unit spends executing the instructions of a certain task, so we understand which model is executed faster by the CPU.
6)Wall time refers to the actual time taken from the start of the process to the end and is measured by a standard clock or timer. We would compare the Wall time to the CPU time to compare the efficiency of each algorithm through the time the CPU takes compared to the full system.
4.3) Naïve Bayes
The Naïve Bayes Algorithm is a machine learning technique that is widely used in text classification and spam filtering[4]. It is a popular algorithm that is based on Bayes' theorem, which is alternatively known as Bayes’ law. It provides a mathematical rule for inverting conditional probabilities, allowing one to find the probability of a cause given its effect[2]. It is a generative learning algorithm that can distribute inputs into sections of a given class or category. The features of the input data have to be conditionally independent of each other in this case, “Spam” and “Ham,” allowing the algorithm to make predictions quickly and accurately. The two Naïve Bayes algorithms that I will be using are Multinomial Naïve Bayes and Bernoulli Naïve Bayes. To test the effectiveness of the Naïve Bayes algorithm, another machine learning algorithm, MLPClassifier, is added.
1) Multinomial Naïve Bayes focuses on the number of times a word would occur in a document or dataset(multinomial distributed)[4]. It is used primarily to solve text classification problems. It does this by categorizing the information into sections. 
2) Bernoulli Naïve Bayes operates similarly to Multinomial Naïve Bayes and is used to categorize documents and datasets too; however, instead of focusing on the frequency of the words, it uses Boolean variables (1 and 0) as its variables. 
3) MLPClassifier is a sort of neural network that can map out data sets to a set of appropriate outputs, similar to the Bernoulli naïve Bayes. It also uses 1’s and 0’s as its variables[7]. It consists of multiple layers that are connected to the preceding one. On each node, there are activation functions except for the nodes that are on the input layer. Between the input and output layers, there must be one or more nonlinear hidden layers.

4.4) TfidfVectorizer
TfidfVectorizer is a popular feature extraction technique used in natural language processing (NLP). Its goal is to turn text into numerical data that machine learning algorithms can work with. It is split up into three parts:
1)Term Frequency(TF) - This measures how frequently a word appears in the text. The premise being that if the word appears frequently, then it must be important. However, this by itself isn’t enough, as it would pick up lexical categories such as nouns as important.
2) Inverse document frequency(IDF) – This measures how frequently a word appears in all texts in a data set. What separates this from term frequency is that the more texts the word appears in, the less important it is.
3)TF-IDF – Is the product of TF and IDF combined. As a result, the algorithms can detect the most important words without having to worry about the lexical categories.

5)	 Results





Fig3. Results of all the algorithms on the dataset.




Fig4. Confusion matrix for all the models

All algorithms performed significantly well with the spam detection, with almost perfect Accuracy and Precision scores. MLPClassifier is the most effective in accuracy with 0.9794%. Precision Multinomial takes pole position with an impressive perfect score of 1. MLPClassifier leads in Recall and F1 score, indicating its results are superior in the quality of detection. On the other hand, the naïve Bayes models operate on significantly faster CUP and Wall times, showing they lack slightly in quality; they are more efficient for systems to run. The confusion matrix (Fig.) gives us an insight into the TP, TN, FP, and FN values each algorithm produces and uses to create its results.
6)	Conclusion
To conclude, there are many ways for cybercriminals to conduct phishing attacks on users. It is important to educate users on the tactics used as they are becoming more advanced day by day. Email spam detection can be conducted through natural language processing (NLP)  with Machine learning algorithms, in this case, Naïve Bayes and MLP classifiers. Researchers are working constantly to fight the cybercriminals who plague users' inboxes. Gmail and Yahoo both use machine learning techniques in their filtering systems to provide a safe environment for users. Each machine learning algorithm is effective in its own regard, with others excelling in an area more than others. The user's or the company's preference depends on whether they prioritize quality over execution time.

           7)    References
1.	Younas, Z. (2024). Email spam detection dataset. Kaggle.
2.	Vohra, H. Kumar, M. Email Spam Detection Using Naïve Bayes. Department of Computer Science & Engineering, Delhi Technological University, India. XISDXJXSU Journal, 19(4). 
3.	Sulthana, R. Verma, A. Jaithubi, A.K. A detailed analysis of spam emails and detection using Machine Learning algorithms. Department of Computing and Mathematical Sciences, University of Greenwich. Department of Computer Science, Birla Institute of Technology & Science, Pilani-Dubai Campus. Department of Computer Science, RMD Engineering College, Kavaraipettai, Tamil Nādu, India.
4.	Ray, S. (2017, September). Naive Bayes Classifier Explained with Practical Problems. Analytics Vidhya.
5.	Harikrishnan, NB. (2019, June). Confusion matrix, accuracy, precision, recall, F1 score. Medium. 
6.	GeeksforGeeks. (2025). Essential metrics for model assessment: TP, TN, FP, FN in machine learning. (ALL TP, TN, FP, FN calculation pictures) 
7.	Fuchs, M. (2021). NN multi-layer perceptron classifier (MLPClassifier). 
8.	scikit-learn developers. (n.d.). TfidfVectorizer. scikit-learn. 
9.	GeeksforGeeks. (2025). How to store a TfidfVectorizer for future use in scikit-learn. 
10.	Kontorskyy,D. (2024). Outlook Spam Filter Explained: A Step-by-Step Guide. Mailtrap.
11.	Morelo,  D. (2025). Gmail spam filter: How it works and how to customize it. Clean Email
12.	Kumaran, N. (2023). New Gmail protections for a safer, less spammy inbox. Google. 
13.	Ferrara, E. (2004). The history of digital spam. Communications of the ACM, 47(8), 76–82. 
14.	Ghozali, A. Nozari, H. Zamzuri, F. Types and Methods of Managing SPAM Messages: A Review. Faculty of Computer and Mathematical Sciences. Universiti of Teknologi Mara, Malaysia.
15.	Taylor, J. (2024). The history of email spam. Knak.
16.	Ferrara, E.(2019). The History of Digital Spam. Communications of the ACM.
17.	All figs are made by me.

