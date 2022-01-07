# -*- coding: utf-8 -*-
#Programma 1 - Vincenzo Sammartino (mat. 599203)

import sys
import nltk
from nltk import bigrams

def CalcolaLunghezzaTokens(frasi): #Calcola il numero dei token totali del file
    lunghezzaTotale=0.0
    for frase in frasi:
        tokens = nltk.word_tokenize(frase)
        lunghezzaTotale = lunghezzaTotale + len(tokens)
    return lunghezzaTotale
    
def CalcolaLunghezzaMediaFrasiTokens(frasi): #Calcola la lunghezza media delle frasi in termini di tokens
    numeroFrasi = 0.0
    numeroTokensPerFrase = 0.0
    for frase in frasi:
        numeroFrasi = numeroFrasi + 1
        tokens = nltk.word_tokenize(frase)
        numeroTokensPerFrase = numeroTokensPerFrase + len(tokens)
    LunghezzaMediaFrase = numeroTokensPerFrase/numeroFrasi
    return LunghezzaMediaFrase
    
def CalcolaLunghezzaMediaCaratteri(frasi): #Calcola la lunghezza media delle parole in termini di caratteri
    NumeroTokens = 0.0
    LunghezzaCaratteri = 0.0
    for frase in frasi:
        tokens = nltk.word_tokenize(frase)
        for tok in tokens:
            NumeroTokens = NumeroTokens + 1
            LunghezzaCaratteri = LunghezzaCaratteri + len(tok)

    LunghezzaMediaCaratteri = LunghezzaCaratteri/NumeroTokens
    return LunghezzaMediaCaratteri

def CalcolaGrandezzaVocabolario5000(frasi): #Calcola la grandezza del vocabolario nei primi 5000 tokens
    vocabolario = []
    corpus = []
    for frase in frasi:
        tokens = nltk.word_tokenize(frase)
        corpus = corpus + tokens
    corpus5000 = corpus[0:5000]
    vocabolario = list(set(corpus5000))
    return len(vocabolario)

def CalcolaDistribuzioneClasseFreq(frasi,n): #Stampa la distribuzione delle classi di frequenza Vn in porzioni incrementali di 500 tokens
    corpus = []
    for frase in frasi:
       tokens = nltk.word_tokenize(frase)
       corpus = corpus + tokens
    for i in range(500, len(corpus), 500): #La i parte da 500 e aumenta per intervalli di 500, per tutta la lunghezza del corpus
        porzioneCorpus = corpus[0:i] #Seleziono la porzione del corpus da 0 a i
        classeFrequenza = [] #Azzero il vettore per ogni classe di frequenza
        for tok in porzioneCorpus:
            if porzioneCorpus.count(tok) == n:
                classeFrequenza.append(tok)
                
        classeFrequenza = list(set(classeFrequenza)) #Uso la funzione set per evitare ripetizioni della stessa parola
        print("Nei primi", i , "tokens ci sono", len(classeFrequenza), "token di classe V", n)

def AnnotazioneLinguistica(frasi): #Esegue la POS delle frasi che gli vengono passate come input
	tokensPOStot = []
	tokensTOT = []
	for frase in frasi:
		tokens = nltk.word_tokenize(frase)
		tokensPOS = nltk.pos_tag(tokens)
		tokensPOStot = tokensPOStot + tokensPOS
	return tokensPOStot

def CalcolaMediaSostantiviFrase(frasi): #Calcola il numero medio di sostantivi per frase
    numeroFrasi = 0.0
    numeroSostantivi = 0.0
    tokensPOStot = []
    for frase in frasi:
        tokens = nltk.word_tokenize(frase)
        tokensPOS = nltk.pos_tag(tokens)
        tokensPOStot = tokensPOStot + tokensPOS
        numeroFrasi = numeroFrasi + 1
    for tok in tokensPOStot:
        if tok[1] == "NN" or tok[1] == "NNP" or tok[1] == "NNS" or tok[1] == "NNPS":
            numeroSostantivi = numeroSostantivi + 1

    MediaSostantiviFrase = numeroSostantivi / numeroFrasi
    return MediaSostantiviFrase

def CalcolaMediaVerbiFrase(frasi): #Calcola il numero medio di verbi per frase
    numeroFrasi = 0.0
    numeroVerbi = 0.0
    tokensPOStot = []
    for frase in frasi:
        tokens = nltk.word_tokenize(frase)
        tokensPOS = nltk.pos_tag(tokens)
        tokensPOStot = tokensPOStot + tokensPOS
        numeroFrasi = numeroFrasi + 1
    for tok in tokensPOStot:
        if tok[1] == "VB" or tok[1] == "VBD" or tok[1] == "VBG" or tok[1] == "VBN" or tok[1] == "VBP" or tok[1] == "VBZ":
            numeroVerbi = numeroVerbi + 1

    MediaVerbiFrase = numeroVerbi / numeroFrasi
    return MediaVerbiFrase
        
def CalcolaDensitàLessicale(TestoPOS): #Calcola la densità lessicale del testo precedentemente sottoposto a POS tagging
    sostantivi = 0
    verbi = 0
    avverbi = 0
    aggettivi = 0
    punteggiatura = 0
    for bigramma in TestoPOS:
        if bigramma[1] == "NN" or bigramma[1] == "NNS" or bigramma[1] == "NNP" or bigramma[1] == "NNPS":
            sostantivi = sostantivi + 1
        elif bigramma[1] == "VB" or bigramma[1] == "VBD" or bigramma[1] == "VBG" or bigramma[1] == "VBN" or bigramma[1] == "VBP" or bigramma[1] == "VBZ":
            verbi = verbi + 1
        elif bigramma[1] == "RB" or bigramma[1] == "RBR" or bigramma[1] == "RBS":
            avverbi = avverbi + 1
        elif bigramma[1] == "JJ" or bigramma[1] == "JJR" or bigramma[1] == "JJS":
            aggettivi = aggettivi + 1
        elif bigramma[1] == "." or bigramma[1] == ",":
            punteggiatura = punteggiatura + 1

    TotParole = len(TestoPOS)
    Densità = (sostantivi + verbi + avverbi + aggettivi)/(TotParole - punteggiatura)
    return Densità
    
def main(file1, file2):  #La funzione principale richiama le rispettive funzioni e stampa gli elementi richiesti
    #Leggo i file, li assegno alle variabili e li tokenizzo
    fileInput1 = open(file1, mode="r", encoding="utf-8")
    fileInput2 = open(file2, mode="r", encoding="utf-8")
    raw1 = fileInput1.read()
    raw2 = fileInput2.read()
    tokenizzatore = nltk.data.load('tokenizers/punkt/english.pickle') #Carico il tokenizzatore in lingua inglese

    frasi1 = tokenizzatore.tokenize(raw1)
    frasi2 = tokenizzatore.tokenize(raw2)
    
    lunghezzatok1 = round(CalcolaLunghezzaTokens(frasi1)) #Uso la funzione round per visualizzare il numero intero senza decimali
    lunghezzatok2 = round(CalcolaLunghezzaTokens(frasi2))
    
    lunghezzafrasi1 = len(frasi1)
    lunghezzafrasi2 = len(frasi2)

    print("\t\t--Progetto di Vincenzo Sammartino (matricola 599203)--")
    print()
    print("Programma 1 - Punto n.1 --CALCOLO TOKEN E FRASI--")

    print("Il file", file1, "ha una lunghezza di", lunghezzatok1, "tokens")
    print("Il file", file2, "ha una lunghezza di", lunghezzatok2, "tokens")

    print()

    if (lunghezzatok1 > lunghezzatok2): #Confronta le statistiche dei file
        print("Il file", file1, "è formato da", lunghezzatok1, "tokens, dunque risulta avere più token del file", file2, "il quale contiene", lunghezzatok2, "tokens.")
    elif  (lunghezzatok1 < lunghezzatok2):
        print ("Il file", file2, "è formato da", lunghezzatok2, "tokens, dunque risulta avere più token del file", file1, "il quale contiene", lunghezzatok1, "tokens.")
    else:
        print("Entrambi i file hanno", lunghezzatok1, "tokens.")

    print()
    
    print("Il file", file1, "è composto da", lunghezzafrasi1, "frasi")
    print("Il file", file2, "è composto da", lunghezzafrasi2, "frasi")

    print()
    
    if (lunghezzafrasi1 > lunghezzafrasi2):
        print("Il file", file1, "è formato da", lunghezzafrasi1, "frasi, dunque ha un numero maggiore di frasi rispetto al file", file2, "il quale contiene", lunghezzafrasi2, "frasi.")
    elif  (lunghezzafrasi1 < lunghezzafrasi2):
        print ("Il file", file2, "è formato da", lunghezzafrasi2, "frasi, dunque ha un numero maggiore di frasi rispetto al file", file1, "il quale contiene", lunghezzafrasi1, "frasi.")
    else:
        print("Entrambi i file sono formati da", lunghezzafrasi1, "frasi.")

    print()
    
    print("Punto n.2\t--LUNGHEZZA MEDIA DELLE FRASI--")
    lunghezzaMediaFrasi1 = CalcolaLunghezzaMediaFrasiTokens(frasi1)
    lunghezzaMediaFrasi2 = CalcolaLunghezzaMediaFrasiTokens(frasi2)
    print("Il file", file1,"ha una lunghezza media di frase di", lunghezzaMediaFrasi1, "tokens")
    print("Il file", file2,"ha una lunghezza media di frase di", lunghezzaMediaFrasi2, "tokens")

    print()
    
    if (lunghezzaMediaFrasi1  > lunghezzaMediaFrasi2):
        print("Nel file", file1, "la lunghezza media di frase è", round(lunghezzaMediaFrasi1), "tokens, che risulta maggiore rispetto al file", file2, "la quale lunghezza media di frase è", round(lunghezzaMediaFrasi2), "tokens") #Uso la funzione round per arrotondare i risultati
    elif (lunghezzaMediaFrasi1  < lunghezzaMediaFrasi2):
        print("la lunghezza media di frase è", round(lunghezzaMediaFrasi2), "tokens, che risulta maggiore del file", file1, "la quale lunghezza media di frase è", round(lunghezzaMediaFrasi1), "tokens")
    else:
        print("Entrambi i file hanno la stessa lunghezza media di frase:", round(lunghezzafrasi1), "tokens") 
    
    print()
    print("\t--LUNGHEZZA MEDIA DI PAROLA IN TERMINI DI CARATTERI--")
    
    lunghezzaMediaParola1=CalcolaLunghezzaMediaCaratteri(frasi1)
    lunghezzaMediaParola2=CalcolaLunghezzaMediaCaratteri(frasi2)
    print("Il file", file1, "ha una lunghezza media di parola di", lunghezzaMediaParola1, "caratteri")
    print("Il file", file2, "ha una lunghezza media di parola di", lunghezzaMediaParola2, "caratteri")

    print()

    if (lunghezzaMediaParola1  > lunghezzaMediaParola2):
        print("Nel file", file1, "la lunghezza media di parola è", round(lunghezzaMediaParola1,2), "caratteri, che risulta maggiore rispetto al file", file2, "la quale lunghezza media di parola è", round(lunghezzaMediaParola2,2), "caratteri")
    elif (lunghezzaMediaParola1  < lunghezzaMediaParola2):
        print("la lunghezza media di parola è", round(lunghezzaMediaParola2,2), "caratteri, che risulta maggiore del file", file1, "la quale lunghezza media di parola è", round(lunghezzaMediaParola1,2), "caratteri")
    else:
        print("Entrambi i file hanno la stessa lunghezza media di parola:", round(lunghezzafrasi1,2), "caratteri")

    print()
    print("Punto n.3 \t--GRANDEZZA DEL VOCABOLARIO NEI PRIMI 5000 TOKEN--")

    GrandezzaVocabolario1=CalcolaGrandezzaVocabolario5000(frasi1)
    GrandezzaVocabolario2=CalcolaGrandezzaVocabolario5000(frasi2)
    print("Il vocabolario del file", file1, "è formato da", GrandezzaVocabolario1, "parole tipo.")
    print("Il vocabolario del file", file2, "è formato da", GrandezzaVocabolario2, "parole tipo.")

    print()

    if (GrandezzaVocabolario1 > GrandezzaVocabolario2):
        print("Il vocabolario del file", file1, "è formato da", GrandezzaVocabolario1, "parole tipo, dunque risulta essere più grande del vocabolario del file", file2, "il quale contiene", GrandezzaVocabolario2, "parole tipo.")
    elif (GrandezzaVocabolario1 < GrandezzaVocabolario2):
        print("Il vocabolario del file", file2, "è formato da", GrandezzaVocabolario2, "parole tipo, dunque risulta essere più grande del vocabolario del file", file1, "il quale contiene", GrandezzaVocabolario1, "parole tipo.")
    else:
        print("I file hanno la stessa grandezza del vocabolario:", GrandezzaVocabolario1, "parole tipo.")

    print()
    print("\t--TYPE TOKEN RATIO NEI PRIMI 5000 TOKEN--")
    Primi5000Token1=5000
    Primi5000Token2=5000
    if(lunghezzatok1<5000): #Se il corpus è minore di 5000 token assumo come valore la lunghezza del corpus
        Primi5000Token1 = lunghezzatok1
    if(lunghezzatok2<5000):
        Primi5000Token2 = lunghezzatok2
    
    TypeTokenRatio1=GrandezzaVocabolario1/Primi5000Token1 #Calcolo la TTR nei primi 5000 tokens
    TypeTokenRatio2=GrandezzaVocabolario2/Primi5000Token2
    
    print("L'indice di ricchezza lessicale (Type Token Ratio) nel file", file1, "è pari a", TypeTokenRatio1)
    print("L'indice di ricchezza lessicale (Type Token Ratio) nel file", file2, "è pari a", TypeTokenRatio2)

    print()

    if (TypeTokenRatio1 > TypeTokenRatio2):
        print("La Type Token Ratio del file", file1, "è pari a", round(TypeTokenRatio1,2), "che risulta maggiore, è dunque lessicalmente più ricco del file", file2, "la quale TTR è pari a", round(TypeTokenRatio2,2))
    elif (TypeTokenRatio1 < TypeTokenRatio2):
        print("La Type Token Ratio del file", file2, "è pari a", round(TypeTokenRatio2,2), "che risulta maggiore, è dunque lessicalmente più ricco del file", file1, "la quale TTR è pari a", round(TypeTokenRatio1,2))
    else:
        print("I due file possiedono lo stesso indice di ricchezza lessicale TTR:", round(TypeTokenRatio1,2))

    print()
    print("Punto n.4 \t--DISTRIBUZIONE DELLE CLASSI DI FREQUENZA V1, V5 E V10--")
    print("Stampo di seguito la distribuzione delle classi di frequenza V1, V5 e V10 all'aumentare del corpus per porzioni incrementali di 500 tokens del file", file1)
    DistribuzioneFrequenza1=CalcolaDistribuzioneClasseFreq(frasi1,1)
    print()
    DistribuzioneFrequenza5=CalcolaDistribuzioneClasseFreq(frasi1,5)
    print()
    DistribuzioneFrequenza10=CalcolaDistribuzioneClasseFreq(frasi1,10)
    print()

    print("Stampo di seguito la distribuzione delle classi di frequenza V1, V5 e V10 all'aumentare del corpus per porzioni incrementali di 500 tokens del file", file2)
    DistribuzioneFrequenza1=CalcolaDistribuzioneClasseFreq(frasi2,1)
    print()
    DistribuzioneFrequenza5=CalcolaDistribuzioneClasseFreq(frasi2,5)
    print()
    DistribuzioneFrequenza10=CalcolaDistribuzioneClasseFreq(frasi2,10)
    print()
    
    print()
    print("Punto n.5 \t--MEDIA DEI SOSTANTIVI PER FRASE--")
    MediaSostantivi1 = CalcolaMediaSostantiviFrase(frasi1)
    MediaSostantivi2 = CalcolaMediaSostantiviFrase(frasi2)
    print("Nel file", file1, "in media ci sono", MediaSostantivi1, "sostantivi per frase")
    print("Nel file", file2, "in media ci sono", MediaSostantivi2, "sostantivi per frase")
    print()
    if (MediaSostantivi1 > MediaSostantivi2):
        print("Il numero medio di sostantivi per frase del file", file1, "è pari a", round(MediaSostantivi1,2), "che risulta maggiore del numero medio di sostantivi per frase del file", file2, "che risulta pari a", round(MediaSostantivi2,2))
    elif (MediaSostantivi1 < MediaSostantivi2):
        print("Il numero medio di sostantivi per frase del file", file2, "è pari a", round(MediaSostantivi2,2), "che risulta maggiore del numero medio di sostantivi per frase del file", file1, "che risulta pari a", round(MediaSostantivi1,2))
    else:
        print("I due file possiedono lo stesso numero medio di sostantivi per frase:", MediaSostantivi1)
    
    print()
    print("\t--MEDIA DEI VERBI PER FRASE--")

    MediaVerbi1 = CalcolaMediaVerbiFrase(frasi1)
    MediaVerbi2 = CalcolaMediaVerbiFrase(frasi2)
    print("Nel file", file1, "in media ci sono", MediaVerbi1, "verbi per frase")
    print("Nel file", file2, "in media ci sono", MediaVerbi2, "verbi per frase")
    print()
    if (MediaVerbi1 > MediaVerbi2):
        print("Il numero medio di verbi per frase del file", file1, "è pari a", round(MediaVerbi1,2), "che risulta maggiore del numero medio di verbi per frase del file", file2, "che risulta pari a", round(MediaVerbi2,2))
    elif (MediaVerbi1 < MediaVerbi2):
        print("Il numero medio di verbi per frase del file", file2, "è pari a", round(MediaVerbi2,2), "che risulta maggiore del numero medio di verbi per frase del file", file1, "che risulta pari a", round(MediaVerbi1,2))
    else:
        print("I due file possiedono lo stesso numero medio di verbi per frase:", MediaVerbi1)
    print()
    
    print("Punto n.6 \t--DENSITA' LESSICALE--")
    TestoPOS1 = AnnotazioneLinguistica(frasi1)
    TestoPOS2 = AnnotazioneLinguistica(frasi2)
    DensitàLessicale1 = CalcolaDensitàLessicale(TestoPOS1)
    DensitàLessicale2 = CalcolaDensitàLessicale(TestoPOS2)
    print("La densità lessicale del file", file1, "è pari a", DensitàLessicale1)
    print("La densità lessicale del file", file2, "è pari a", DensitàLessicale2)
    print()
    if (DensitàLessicale1 > DensitàLessicale2):
        print("La densità lessicale del file", file1, "è pari a", round(DensitàLessicale1,2), "che risulta maggiore della densità lessicale del file", file2, "che risulta pari a", round(DensitàLessicale2,2))
    elif (DensitàLessicale1 < DensitàLessicale2):
        print("La densità lessicale del file", file2, "è pari a", round(DensitàLessicale2,2), "che risulta maggiore della densità lessicale del file", file1, "che risulta pari a", round(DensitàLessicale1,2))
    else:
        print("I due file possiedono lo stesso valore di densità lessicale:", DensitàLessicale1)
    
main(sys.argv[1], sys.argv[2])
