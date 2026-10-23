-------------------
GENERAL INFORMATION
-------------------


1. Title of Dataset 
Semantic Relatedness and Similarity Reference Standards for Medical Terms

2. Author Information
  Principal Investigator Contact Information
        Name: Pakhomov, Serguei
           Institution: University of Minnesota
           Address: Institute for Health Informatics
           Email: [email removed]

3. Information about funding sources that supported the collection of the data:
NIH National Library of Medicine R01 grant (LM009623)


--------------------------
SHARING/ACCESS INFORMATION
-------------------------- 


1. Licenses/restrictions placed on the data:
CC0 1.0 Universal 
http://creativecommons.org/publicdomain/zero/1.0/

2. Links to publications that cite or use the data:
Mayo Medical Coders Set (MayoSRS and MiniMayoSRS): Measures of semantic similarity and relatedness in the biomedical domain. Pedersen T., Pakhomov S.V.S., Patwardhan S., and Chute C.G. Journal of Biomedical Informatics. 2007;40(3):288-299.

Mayo Medical Coders Set (MayoSRS and MiniMayoSRS): UMLS-Interface and UMLS-Similarity : Open source software for measuring paths and semantic similarity. McInnes B.T., Pedersen T., and Pakhomov S.V. Proceedings of the Annual Symposium of the American Medical Informatics Association. San Fransisco, CA. 2009;431-435.

UMN Medical Residents Similarity/Relatedenss Set (UMNSRS-Similarity and UMNSRS-Relatedenss): Semantic Similarity and Relatedness between Clinical Terms: An Experimental Study. Pakhomov S., McInnes, B., Adams, T., Liu, Y., Pedersen, T. and Melton, G.B. Proceedings of the Annual Symposium of the American Medical Informatics Association. Washington, D.C. November, 2010.

Modified Medical Residents Similarity/Relatedness Set (UMNSRS-Similarity-mod and UMNSRS-Relatedenss-mod): Corpus Domain Effects on Distributional Semantic Modeling of Medical Terms. Serguei V.S. Pakhomov, Greg Finley, Reed McEwan, Yan Wang, and Genevieve B. Melton. Bioinformatics. 2016; 32(23):3635-3644.

Towards a Framework for Developing Semantic Relatedness Reference Standards. Pakhomov, Serguei V.S. and Pedersen, Ted and McInnes, Bridget and Melton, Genevieve B. and Ruggieri, Alexander and Chute, Christopher G. J. of Biomedical Informatics. 44 (2): 251-265.

3. Recommended citation for the data:
Pakhomov, Serguei. (2018). Semantic Relatedness and Similarity Reference Standards for Medical Terms. Retrieved from the University of Minnesota Digital Conservancy, http://hdl.handle.net/11299/196265.


---------------------
DATA & FILE OVERVIEW
---------------------


1. File List
   A. Filename: MayoSRS.csv       
      Short description: A set of 101 medical concept pairs manually rater by medical coders for semantic relatedness.        
        
   B. Filename: MiniMayoSRS.csv       
      Short description: A subset of 29 medical concept pairs manually rater by medical coders for semantic relatedness with high inter-rater agreement.       
        
   C. Filename: UMNSRS_similarity.csv       
      Short description: A set of 566 UMLS concept pairs manually rated for semantic similarity using a continuous response scale.

   D. Filename: UMNSRS_relatedenss.csv       
      Short description: A set of 588 UMLS concept pairs manually rated for semantic relatedness using a continuous response scale.

   E. Filename: UMNSRS_similarity_mod449_word2vec.csv      
      Short description: Modification of the UMNSRS-Similarity dataset to exclude control samples and those pairs that did not match text in clinical, biomedical and general English corpora. Exact modifications are detailed in the referenced paper. The resulting dataset contains 449 pairs.

   F. Filename: UMNSRS_relatedness_mod458_word2vec.csv      
      Short description: Modification of the UMNSRS-Similarity dataset to exclude control samples and those pairs that did not match text in clinical, biomedical and general English corpora. Exact modifications are detailed in the paper below. The resulting dataset contains 458 pairs.


-----------------------------------------
DATA-SPECIFIC INFORMATION FOR: MayoSRS.csv
-----------------------------------------
1. Number of variables: 5

2. Number of cases/rows: 101

3. Variable List
    A. Name: Mean
       Description: average rating (Likert scale 1-10) of semantic relatedness between a pair of terms

    B. Name: CUI1 
       Description: the concept unique identifier (CUI) from the Unified Medical Language System Metathesaurus of the first term               
    
    C. Name: CUI2
       Description: the concept unique identifier (CUI) from the Unified Medical Language System Metathesaurus of the second term

    D. Name: TERM1
       Description: the first term presented to subject for rating
 
    E. Name: TERM2
       Description: the second term presented to subject for rating

NOTE: both TERM1 and TERM2 were presented to subjects in a spreadsheet all at the same time. The indices 1 and 2 refer to relative position of the terms in the spreadsheet in a left-to-right order
-----------------------------------------
DATA-SPECIFIC INFORMATION FOR: MiniMayoSRS.csv
-----------------------------------------
1. Number of variables: 6

2. Number of cases/rows: 29

3. Variable List
    A. Name: Physicians
       Description: average rating (Likert scale 1-10) of semantic relatedness between a pair of terms given by physicians

    B. Name: Coders 
       Description: average rating (scale 1-10) of semantic relatedness between a pair of terms given by medical coders            

    C. Name: CUI1
       Description: the concept unique identifier (CUI) from the Unified Medical Language System Metathesaurus of the first term

    D. Name: CUI2
       Description: the concept unique identifier (CUI) from the Unified Medical Language System Metathesaurus of the second term
 
    E. Name: TERM1
       Description: the first term presented to subject for rating

    F. Name: TERM2
       Description: the second term presented to subject for rating

NOTE: both TERM1 and TERM2 were presented to subjects in a spreadsheet all at the same time. The indices 1 and 2 refer to relative position of the terms in the spreadsheet in a left-to-right order
-----------------------------------------
DATA-SPECIFIC INFORMATION FOR: UMNSRS_similarity.csv and UMNSRS_similarity_mod449_word2vec.csv
-----------------------------------------
1. Number of variables: 6, 6

2. Number of cases/rows: 566, 449

3. Variable List
    A. Name: Mean
       Description: average rating of semantic similarity between a pair of terms measured as the position of the finger touch on a bar displayed on a computer screen (offset from the left edge of a computer screen on the x-coordinate). VAS scale 0-1600: 0-least similar, 1600 - most similar.

    B. Name: Stdev 
       Description: standard deviation of the Mean semantic similarity ratings        

    C. Name: CUI1
       Description: the concept unique identifier (CUI) from the Unified Medical Language System Metathesaurus of the first term

    D. Name: CUI2
       Description: the concept unique identifier (CUI) from the Unified Medical Language System Metathesaurus of the second term
 
    E. Name: TERM1
       Description: the first term presented to subject for rating

    F. Name: TERM2
       Description: the second term presented to subject for rating
-----------------------------------------
DATA-SPECIFIC INFORMATION FOR: UMNSRS_relatedness.csv and UMNSRS_relatedness_mod458_word2vec.csv
-----------------------------------------
1. Number of variables: 6

2. Number of cases/rows: 587, 458

3. Variable List
    A. Name: Mean
       Description: average rating of semantic relatedness between a pair of terms measured as the position of the finger touch on a bar displayed on a computer screen (offset from the left edge of a computer screen on the x-coordinate). VAS scale 0-1600: 0-unrelated, 1600 - closely related.

    B. Name: Stdev 
       Description: standard deviation of the Mean semantic similarity ratings       

    C. Name: CUI1
       Description: the concept unique identifier (CUI) from the Unified Medical Language System Metathesaurus of the first term

    D. Name: CUI2
       Description: the concept unique identifier (CUI) from the Unified Medical Language System Metathesaurus of the second term
 
    E. Name: TERM1
       Description: the first term presented to subject for rating

    F. Name: TERM2
       Description: the second term presented to subject for rating