# -*- coding: utf-8 -*-
#Programma 2 - Vincenzo Sammartino (mat. 599203)

import sys
import math
import nltk
from nltk import bigrams

def AnnotazioneLinguistica(frasi): #Esegue il POS tagging delle frasi che gli vengono passate come input
	tokensPOStot = []
	tokensTOT = []
	for frase in frasi:
		tokens = nltk.word_tokenize(frase)
		tokensPOS = nltk.pos_tag(tokens)
		tokensPOStot = tokensPOStot + tokensPOS
	return tokensPOStot

def EstraiBigrammiPOS(TestoAnalizzatoPOS): #Estraggo i bigrammi (token, POS) del testo
    BigrammiEstratti = []
    bigrammaTokPOS = bigrams(TestoAnalizzatoPOS)
    for bigramma in TestoAnalizzatoPOS:
        BigrammiEstratti.append(bigramma)
    return BigrammiEstratti

def CreaDizionarioBigrammi(Bigrammi):
    Dizionario = {}
    ParoleTipo = set(Bigrammi) #Estraggo solo le parole diverse grazie alla funzione set
    for bigramma in Bigrammi:
        frequenza = Bigrammi.count(bigramma)
        Dizionario[bigramma] = frequenza #Assegno ad ogni bigramma la relativa frequenza
    return Dizionario

def Ordina(Dizionario): #Ordina in maniera decrescente il dizionario
    return sorted(Dizionario.items(), key=lambda x: x[1], reverse=True)
       
def DieciPOSFreq (DizionarioOrdinatoBigrammi): #Estraggo le dieci POS più frequenti nel testo
    contatore = 1
    for bigramma in DizionarioOrdinatoBigrammi:
        print("La", contatore, "°", "PoS più frequente è: \"" + bigramma[0][0] + "\"\t", "con frequenza", bigramma[1])
        contatore = contatore + 1
        if contatore > 10:
            break

def VentiSostantiviFreq (DizionarioOrdinatoBigrammi): #Trovo e stampo i venti verbi più frequenti
    sostantiviPOS = []
    contatore = 1
    for bigramma in DizionarioOrdinatoBigrammi:
        if bigramma[0][1] == "NN" or bigramma[0][1] == "NNP" or bigramma[0][1] == "NNS" or bigramma[0][1] == "NNPS":
            sostantiviPOS.append(bigramma)
            print("Il", contatore, "°", "sostantivo più frequente è: \"" + bigramma[0][0] + "\"\t", "con frequenza", bigramma[1])
            contatore = contatore + 1
            if contatore > 20:
                break

def VentiVerbiFreq (DizionarioOrdinatoBigrammi): #Trovo e stampo i venti verbi più frequenti
    verbiPOS = []
    contatore = 1
    for bigramma in DizionarioOrdinatoBigrammi:
        if bigramma[0][1] == "VB" or bigramma[0][1] == "VBD" or bigramma[0][1] == "VBN" or bigramma[0][1] == "VBG" or bigramma[0][1] == "VBP" or bigramma[0][1] == "VBZ":
            verbiPOS.append(bigramma)
            print("Il", contatore, "°", "verbo più frequente è: \"" + bigramma[0][0] + "\"\t", "con frequenza", bigramma[1])
            contatore = contatore + 1
            if contatore > 20:
                break

def VentiSostantiviVerbi (TestoAnalizzatoPOS):
    SostVerbiPOS = []
    DizionarioBigrammi = {}
    BigrammiTestoPOS = list(bigrams(TestoAnalizzatoPOS)) #Creo la lista dei bigrammi (token, POS)
    for bigramma in BigrammiTestoPOS:
        if (bigramma[0][1] == "NN" or bigramma[0][1] == "NNP" or bigramma[0][1] == "NNPS" or bigramma[0][1] == "NNS") and (bigramma[1][1] == "VB" or bigramma[1][1] == "VBG" or bigramma[1][1] == "VBD" or bigramma[1][1] == "VBN" or bigramma[1][1] == "VBZ" or bigramma[1][1] == "VBP"):
            tupla = (bigramma[0][0], bigramma[1][0])
            SostVerbiPOS.append(tupla)
            frequenza = SostVerbiPOS.count(tupla) #Calcolo la frequenza della tupla
            DizionarioBigrammi[tupla] = frequenza  #Creo un dizionario e assegno ad ogni tupla la relativa frequenza

    DizionarioBigrammi = Ordina(DizionarioBigrammi) #Ordino il dizionario in modo decrescente
    contatore = 1
    for bigramma in DizionarioBigrammi: #Stampo le venti coppie con frequenza maggiore
        print("La", contatore, "°", "coppia più frequente è: \"" + bigramma[0][0], bigramma[0][1] + "\"\t", "con frequenza", bigramma[1])
        contatore = contatore + 1
        if contatore > 20:
            break

def VentiAggettiviSostantivi (TestoAnalizzatoPOS):
    SostVerbiPOS = []
    DizionarioBigrammi = {}
    BigrammiTestoPOS = list(bigrams(TestoAnalizzatoPOS))
    for bigramma in BigrammiTestoPOS:
        if (bigramma[0][1] == "JJ" or bigramma[0][1] == "JJS" or bigramma[0][1] == "JJR") and (bigramma[1][1] == "NN" or bigramma[1][1] == "NNP" or bigramma[1][1] == "NNPS" or bigramma[1][1] == "NNS"):
            tupla = (bigramma[0][0], bigramma[1][0]) #Seleziono il bigramma
            SostVerbiPOS.append(tupla)
            frequenza = SostVerbiPOS.count(tupla)
            DizionarioBigrammi[tupla] = frequenza  #Creo un dizionario e assegno ad ogni bigramma la relativa frequenza

    DizionarioBigrammi = Ordina(DizionarioBigrammi) #Ordino attraverso l'apposita funzione il dizionario in modo decrescente
    contatore = 1
    for bigramma in DizionarioBigrammi: #Stampo le 20 coppie con maggior frequenza
        print("La", contatore, "°", "coppia più frequente è: \"" + bigramma[0][0], bigramma[0][1] + "\"\t", "con frequenza", bigramma[1])
        contatore = contatore + 1
        if contatore > 20:
            break

def CalcolaProbCongiunta(frasi):
    lunghezzaTotale = 0.0
    Dizionario = {}
    tokens = []
    for frase in frasi:
        tokens = tokens + nltk.word_tokenize(frase)
        lunghezzaTotale = lunghezzaTotale + len(tokens)
    bigrammi = list(bigrams(tokens))
    BigrammiDiversi = (list(set(bigrammi)))
    for bigramma in BigrammiDiversi:
        token1 = bigramma[0]
        token2 = bigramma[1]
        freqToken1 = tokens.count(token1)
        freqToken2 = tokens.count(token2)
        if(freqToken1>3 and freqToken2>3):
            freqBigramma1 = bigrammi.count(bigramma)
            probCondizionata = freqBigramma1 / freqToken1
            probToken1 = freqToken1 / len(tokens)
            probCongiunta = probCondizionata * probToken1 #Calcolo la probabilità congiunta
            Dizionario[bigramma] = probCongiunta  #Creo un dizionario e assegno ad ogni bigramma la relativa probabilità congiunta

    Dizionario = Ordina(Dizionario) #Ordino in modo decrescente il dizionario
    contatore = 1
    for elemento in Dizionario:
        print("Il", contatore, "°", "bigramma \"" + elemento[0][0], elemento[0][1] + "\" con probabilità congiunta pari a", elemento[1])
        contatore = contatore + 1
        if contatore == 21:
            break

def CalcolaProbCondizionata(frasi):
    lunghezzaTotale = 0.0
    Dizionario = {}
    tokens = []
    for frase in frasi:
        tokens = tokens + nltk.word_tokenize(frase)
        lunghezzaTotale = lunghezzaTotale + len(tokens)
    bigrammi = list(bigrams(tokens))
    BigrammiDiversi = (list(set(bigrammi)))
    for bigramma in BigrammiDiversi:
        token1 = bigramma[0]
        token2 = bigramma[1]
        freqToken1 = tokens.count(token1)
        freqToken2 = tokens.count(token2)
        if(freqToken1>3 and freqToken2>3):
            freqBigramma1 = bigrammi.count(bigramma)
            probCondizionata = freqBigramma1 / freqToken1
            Dizionario[bigramma] = probCondizionata #Creo un dizionario e assegno ad ogni bigramma la relativa probabilità condizionata
            
    Dizionario = Ordina(Dizionario) #Ordino in maniera decrescente il dizionario
    contatore = 1
    for elemento in Dizionario: #Stampo solo i primi 20 elementi in ordine di frequenza
        print("Il", contatore, "°", "bigramma \"" + elemento[0][0], elemento[0][1] + "\" con probabilità condizionata pari a", elemento[1])
        contatore = contatore + 1
        if contatore == 21:
            break

def CalcolaLMI(frasi):
    lunghezzaTotale = 0.0
    Dizionario = {}
    tokens = []
    for frase in frasi:
        tokens = tokens + nltk.word_tokenize(frase)
        lunghezzaTotale = lunghezzaTotale + len(tokens)
    bigrammi = list(bigrams(tokens))
    BigrammiDiversi = (list(set(bigrammi)))
    for bigramma in BigrammiDiversi:
        token1 = bigramma[0]
        token2 = bigramma[1]
        freqToken1 = tokens.count(token1)
        freqToken2 = tokens.count(token2)
        if(freqToken1>3 and freqToken2>3):
            freqBigramma1 = bigrammi.count(bigramma)
            probToken1 = freqToken1 / len(tokens) #Calcolo la probabilità del token
            probToken2 = freqToken2 / len(tokens)
            probBigramma1 = freqBigramma1/len(tokens) #Calcolo la probabilità del bigramma
            LMI = freqBigramma1 * math.log((probBigramma1/(probToken1*probToken2)),2) #Calcolo la LMI seguendo la formula
            Dizionario[bigramma] = LMI #Creo un dizionario con i bigrammi e le relative LMI

    Dizionario = Ordina(Dizionario) #Ordino il dizionario in modo decrescente
    contatore = 1
    for elemento in Dizionario: #Stampo solo i primi 20 elementi in ordine di frequenza decrescente
        print("Il", contatore, "°", "bigramma \"" + elemento[0][0], elemento[0][1] + "\" con LMI pari a", elemento[1])
        contatore = contatore + 1
        if contatore == 21:
            break


def Calcola15NomiFreq(NamedEntity): #Stampa i 15 nomi in ordine di frequenza
    ListaNomi = []
    for nodo in NamedEntity:
        NE=''
        if hasattr(nodo, 'label'):
            if nodo.label() in ["PERSON"]:
                for partNE in nodo.leaves():
                    NE = NE + partNE[0]
                ListaNomi.append(NE)
    DistFrequenza = nltk.FreqDist(ListaNomi)
    DistFrequenza = DistFrequenza.most_common(15)
    contatore = 1
    for nome in DistFrequenza:
        print("Il", contatore, "°", "nome proprio di persona più frequente è \"" +  nome[0] + "\"\t", "con frequenza", nome[1])
        contatore = contatore + 1

def Calcola15LuoghiFreq(NamedEntity): #Stampa i 15 luoghi in ordine di frequenza
    ListaLuoghi = []
    for nodo in NamedEntity:
        NE=''
        if hasattr(nodo, 'label'):
            if nodo.label() in ["GPE"]:
                for partNE in nodo.leaves():
                    NE = NE + partNE[0]
                ListaLuoghi.append(NE)
    DistFrequenza = nltk.FreqDist(ListaLuoghi) #Creo la distribuzione di frequenza della lista contenente le Entità Nominate
    DistFrequenza = DistFrequenza.most_common(15) #Seleziono i 15 luoghi con maggiore frequenza
    contatore = 1
    for nome in DistFrequenza:
        print("Il", contatore, "°", "nome proprio di luogo più frequente è \"" + nome[0] + "\"\t", "con frequenza", nome[1])
        contatore = contatore + 1

def CalcolaMarkovOrdine1(frasi): 
    lunghezzaTotale = 0.0
    tokens = []
    probMAX = 0
    for frase in frasi:
        tokens = tokens + nltk.word_tokenize(frase)
        lunghezzaTotale = lunghezzaTotale + len(tokens)
        vocabolario = len(set(tokens))
        frase1 = nltk.word_tokenize(frase)
        if(len(frase1)>8 and len(frase1)<15):
            frase1 = nltk.word_tokenize(frase)
            bigrammi = list(bigrams(frase1))
            probabilità = 1
            for bigramma in bigrammi:
                token1 = bigramma[0]
                token2 = bigramma[1]
                freqToken1 = tokens.count(token1)
                freqToken2 = tokens.count(token2)
                freqBigramma1 = bigrammi.count(bigramma)
                probabilità = probabilità * ((freqBigramma1 + 1) / (freqToken1 + vocabolario)) #Calcolo la p smoothed dei bigrammi della frase
            probabilità = probabilità * ((freqToken1 + 1) / (len(tokens) + vocabolario)) #Moltiplico la probabilità ottenuta per la probabilità del primo bigramma (bigramma[0])
            if probabilità > probMAX: #Trovo la frase con probabilità maggiore
                probMAX = probabilità
                FraseMAX = frase

    print("La frase: \"" + FraseMAX + "\" ha probabilità:", probMAX)
            
def main(file1, file2):   #La funzione principale richiama le rispettive funzioni e stampa gli elementi richiesti
    #Leggo i file, li assegno alle variabili e li tokenizzo
    fileInput1 = open(file1, mode="r", encoding="utf-8")
    fileInput2 = open(file2, mode="r", encoding="utf-8")
    
    raw1 = fileInput1.read()
    raw2 = fileInput2.read()
    tokenizzatore = nltk.data.load('tokenizers/punkt/english.pickle') #Carico il tokenizzatore in lingua inglese

    frasi1 = tokenizzatore.tokenize(raw1)
    frasi2 = tokenizzatore.tokenize(raw2)
    
    TestoAnalizzatoPOS1 = AnnotazioneLinguistica(frasi1)
    TestoAnalizzatoPOS2 = AnnotazioneLinguistica(frasi2)

    BigrammiPOS1 = EstraiBigrammiPOS(TestoAnalizzatoPOS1)
    BigrammiPOS2 = EstraiBigrammiPOS(TestoAnalizzatoPOS2)

    DizionarioBigrammi1 = CreaDizionarioBigrammi(BigrammiPOS1)
    DizionarioBigrammi2 = CreaDizionarioBigrammi(BigrammiPOS2)

    DizionarioOrdinatoBigrammi1 = Ordina(DizionarioBigrammi1)
    DizionarioOrdinatoBigrammi2 = Ordina(DizionarioBigrammi2)

    print("\t\t--Progetto di Vincenzo Sammartino (matricola 599203)--")
    print()
    print("Programma 2 - Punto n.1 \t ESTRAZIONE DELLE 10 POS PIU' FREQUENTI")
    print("Elenco di seguito le 10 Part of Speech più frequenti del file", file1)
    DieciPOS1 = DieciPOSFreq(DizionarioOrdinatoBigrammi1)
    print()
    print("Elenco di seguito le 10 Part of Speech più frequenti del file", file2)
    DieciPOS2 = DieciPOSFreq(DizionarioOrdinatoBigrammi2)
    print()

    print("\t ESTRAZIONE DEI 20 SOSTANTIVI PIU' FREQUENTI")
    print("Elenco di seguito i venti sostantivi più frequenti del file", file1)
    SostantiviFreq1 = VentiSostantiviFreq(DizionarioOrdinatoBigrammi1)
    print()
    print("Elenco di seguito i venti sostantivi più frequenti del file", file2)
    SostantiviFreq2 = VentiSostantiviFreq(DizionarioOrdinatoBigrammi2)
    print()

    print("\t ESTRAZIONE DEI 20 VERBI PIU' FREQUENTI")
    print("Elenco di seguito i venti verbi più frequenti del file", file1)
    VerbiFreq1 = VentiVerbiFreq(DizionarioOrdinatoBigrammi1)
    print()
    print("Elenco di seguito i venti verbi più frequenti del file", file2)
    VerbiFreq2 = VentiVerbiFreq(DizionarioOrdinatoBigrammi2)
    print()
    
    print("\t ESTRAZIONE DEI 20 BIGRAMMI SOSTANTIVO-VERBO PIU' FREQUENTI")
    print("Elenco di seguito i venti bigrammi Sostantivo-Verbo più frequenti del file", file1)
    VentiSostantiviVerbi1 = VentiSostantiviVerbi(BigrammiPOS1)
    print()
    print("Elenco di seguito i venti bigrammi Sostantivo-Verbo più frequenti del file", file2)
    VentiSostantiviVerbi2 = VentiSostantiviVerbi(BigrammiPOS2)

    print()
    print("\t ESTRAZIONE DEI 20 BIGRAMMI AGGETTIVO-SOSTANTIVO PIU' FREQUENTI")
    print("Elenco di seguito i venti bigrammi Aggettivo-Sostantivo più frequenti del file", file1)
    VentiAggettiviSostantivi1 = VentiAggettiviSostantivi(BigrammiPOS1)
    print()
    print("Elenco di seguito i venti bigrammi Aggettivo-Sostantivo più frequenti del file", file2)
    VentiAggettiviSostantivi2 = VentiAggettiviSostantivi(BigrammiPOS2)
    print()

    print("Punto n.2 \t ESTRAZIONE DEI 20 BIGRAMMI CON PROBABILITA' CONGIUNTA MAGGIORE")
    print("Di seguito elenco i primi 20 bigrammi con probabilità congiunta maggiore del file", file1)
    ProbCongiunta1 = CalcolaProbCongiunta(frasi1)
    print()
    print("Di seguito elenco i primi 20 bigrammi con probabilità congiunta maggiore del file", file2)
    ProbCongiunta2 = CalcolaProbCongiunta(frasi2)

    print()
    print(" \t ESTRAZIONE DEI 20 BIGRAMMI CON PROBABILITA' CONDIZIONATA MAGGIORE")
    print("Di seguito elenco i primi 20 bigrammi con probabilità condizionata maggiore del file", file1)
    ProbCondizionata1 = CalcolaProbCondizionata(frasi1)
    print()
    print("Di seguito elenco i primi 20 bigrammi con probabilità condizionata maggiore del file", file2)
    ProbCondizionata2 = CalcolaProbCondizionata(frasi2)

    print()
    print(" \t ESTRAZIONE DEI 20 BIGRAMMI CON FORZA ASSOCIATIVA (LMI) MAGGIORE")
    print("Di seguito elenco i primi 20 bigrammi con forza associativa maggiore del file", file1)
    LMI1 = CalcolaLMI(frasi1)
    print()
    print("Di seguito elenco i primi 20 bigrammi con forza associativa maggiore del file", file2)
    LMI2 = CalcolaLMI(frasi2)

    print()
    print("Punto n.3 \t ESTRAZIONE DELLA FRASE CON PROBABILITÀ MAGGIORE SECONDO IL MODELLO DI MARKOV DI ORDINE 1 (USANDO LO ADD-ONE SMOOTHING)")
    print("Di seguito elenco la frase con maggiore probabilità secondo il modello Markoviano di ordine 1 del file ", file1)
    Frase1 = CalcolaMarkovOrdine1(frasi1)
    print()
    print("Di seguito elenco la frase con maggiore probabilità secondo il modello Markoviano di ordine 1 del file ", file2)
    Frase2 = CalcolaMarkovOrdine1(frasi2)
    print()

    print("Punto n.4 \t ESTRAZIONE DEI 15 NOMI PROPRI DI PERSONA PIU' FREQUENTI")
    NamedEntity1 = nltk.ne_chunk(TestoAnalizzatoPOS1) #Identifico e classifico le Entità Nominate (NE)
    NamedEntity2 = nltk.ne_chunk(TestoAnalizzatoPOS2)

    print("Di seguito elenco i primi 15 nomi propri di persona (PERSON) in ordine di frequenza del file", file1)
    NomiPersonaFreq1 = Calcola15NomiFreq(NamedEntity1)
    print()
    print("Di seguito elenco i primi 15 nomi propri di persona (PERSON) in ordine di frequenza del file", file2)
    NomiPersonaFreq2 = Calcola15NomiFreq(NamedEntity2)

    print()
    print("\t ESTRAZIONE DEI 15 NOMI PROPRI DI LUOGO PIU' FREQUENTI")
    print("Di seguito elenco i primi 15 nomi propri di luogo (GPE) in ordine di frequenza del file", file1)
    NomiLuogoFreq1 = Calcola15LuoghiFreq(NamedEntity1)
    print()
    print("Di seguito elenco i primi 15 nomi propri di luogo (GPE) in ordine di frequenza del file", file2)
    NomiLuogoFreq2 = Calcola15LuoghiFreq(NamedEntity2)

main(sys.argv[1], sys.argv[2])
