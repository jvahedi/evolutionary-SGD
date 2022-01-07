# EvSGD
Gradient descent using evolutionary algorithms to perform batch selection.

Notebook structure provides recommended workflow. Generally, we recommend using test_EASGD() from testing.py for testing. How to run this code:

  1. Set a_true
  2. First generate dataset M using Y() in utils.py (example - M = Y(a=a_true,x=np.array([((20/40)*i-10) for i in range(40)]), s = 0, norm = False))
  3. Run models:
      CEvSGD    -> test_EASGD(M,1000000,4,1)
      EEvSGD    -> test_EASGD(M,1000000,4,2)
      SSGD      -> test_EASGD(M,1000000,4,3)
      SGD       -> test_EASGD(M,1000000,4,4)
      
Users can configure specific model parameters inside of test_EASGD()


NOTE: Notebook has different function names for models as compared to main code & paper, detailed below

(EvSGD)           -> (SSGD)
(EA_SGD)          -> (CEvSGD)
(EA_SGDsinglea)   -> (EEvSGD)



Developed by John Vahedi & Yousef Ahmad, for IEOR 4540: Data Mining at Columbia University with Dr. Krzysztof Choromanski
