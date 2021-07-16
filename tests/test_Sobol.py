import unittest
import Sobol as sbl

class TestSob(unittest.TestCase):

    def setUp(self):
        self.inst = sbl.Sobol()

    def test_sampling_sequence(self):

        self.assertEqual(self.inst.sampling_sequence(2,2,['uniform','uniform'],2)[0][0],0.43599490214200376)
        self.assertEqual(self.inst.sampling_sequence(2,2,['uniform','uniform'],2)[0][1],0.42036780208748903)

        self.assertEqual(self.inst.sampling_sequence(2,2,['uniform','uniform'],2)[4][0],0.025926231827891333)
        self.assertEqual(self.inst.sampling_sequence(2,2,['uniform','uniform'],2)[4][1],0.42036780208748903)

        self.assertEqual(self.inst.sampling_sequence(2,2,['uniform','uniform'],2)[7][0],0.5496624778787091)
        self.assertEqual(self.inst.sampling_sequence(2,2,['uniform','uniform'],2)[7][1],0.6192709663506637)

        self.assertEqual(self.inst.sampling_sequence(2,2,['normal','uniform'],2)[7][0],-2.136196095668454)
        self.assertEqual(self.inst.sampling_sequence(2,2,['uniform','normal'],2)[7][1],-1.2452880866072316)

    def test_indices(self):

        self.assertEqual(self.inst.indices([[9],[3],[1],[8],[1],[1],[2],[1]],2,2)[0][0],-0.2222222222222222)
        self.assertEqual(self.inst.indices([[9],[3],[1],[8],[1],[1],[2],[1]],2,2)[0][1],-0.21296296296296297)


if __name__ == '__main__':
    unittest.main()