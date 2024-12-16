SELECT
   nsfw,
   topk
FROM
   (
      SELECT
         nsfw,
         sample_id AS topk,
         RANK() OVER (
            PARTITION BY nsfw
            ORDER BY
               vec <#> '[-0.016510009765625, 0.016021728515625, 0.0310516357421875, 0.008148193359375, 0.0175628662109375, 0.002460479736328125, 0.0087127685546875, 0.0120086669921875, 0.06341552734375, 0.00652313232421875, 0.018218994140625, 0.02777099609375, 0.0518798828125, -0.006378173828125, -0.052764892578125, 0.0004036426544189453, 0.072509765625, -0.01102447509765625, 0.015167236328125, 0.013916015625, 0.10760498046875, 0.02191162109375, -0.004550933837890625, 0.02081298828125, -0.024261474609375, 0.028106689453125, 0.0662841796875, -0.006519317626953125, 0.0098114013671875, 0.042572021484375, -0.0148773193359375, 0.0205078125, -0.0238494873046875, -0.0006361007690429688, -0.01457977294921875, 0.04962158203125, 0.0012369155883789062, 0.0093994140625, 0.005023956298828125, 0.1170654296875, -0.02593994140625, 0.0313720703125, -0.0196685791015625, -0.0036525726318359375, -0.001354217529296875, -0.30126953125, -0.00830078125, 0.0128021240234375, 0.032196044921875, -0.0350341796875, 0.039825439453125, -0.00331878662109375, 0.01561737060546875, -0.00443267822265625, 0.0400390625, -0.031829833984375, -0.0164794921875, 0.026947021484375, 0.0177001953125, -0.0009918212890625, 0.043060302734375, 0.018218994140625, -0.00821685791015625, -0.07586669921875, 0.0199737548828125, -0.0172271728515625, 0.0450439453125, -0.047607421875, -0.048309326171875, 0.019775390625, -0.037841796875, -0.0166015625, -0.026092529296875, -0.02496337890625, -0.003635406494140625, -0.0126800537109375, -0.0731201171875, -0.052001953125, 0.006011962890625, -0.01519775390625, -0.00424957275390625, -0.034271240234375, -0.0220794677734375, 0.10284423828125, 0.00997161865234375, 0.0019359588623046875, -0.012298583984375, -0.0246429443359375, -0.09991455078125, -0.017181396484375, 0.007572174072265625, -0.0439453125, -0.50048828125, 0.07763671875, -0.00334930419921875, 0.0020694732666015625, 0.040008544921875, -0.0267486572265625, -0.055816650390625, -0.0299072265625, -0.01238250732421875, -0.0020847320556640625, -0.0262603759765625, 0.001373291015625, 0.0634765625, 0.000637054443359375, -0.1094970703125, -0.0265655517578125, 0.01611328125, -0.0166015625, 0.00965118408203125, -0.060577392578125, -0.0016536712646484375, 0.034393310546875, 0.0240631103515625, 0.007232666015625, 0.0184326171875, 0.0292816162109375, 0.0616455078125, 0.02447509765625, -0.0179290771484375, -0.053253173828125, 0.041778564453125, 0.0170745849609375, 0.0150146484375, 0.03155517578125, 0.0092010498046875, -0.0003840923309326172, -0.0009431838989257812, 0.05059814453125, 0.039215087890625, 0.01456451416015625, -0.0005426406860351562, 0.06475830078125, 0.00464630126953125, 0.0092926025390625, 0.0236663818359375, 0.007678985595703125, 0.00308990478515625, -0.0100555419921875, 1.430511474609375e-06, 0.00041866302490234375, -0.005756378173828125, -0.0015611648559570312, 0.000518798828125, 0.0352783203125, -0.02752685546875, 0.032501220703125, -0.0282135009765625, -0.00390625, 0.0362548828125, -0.0186614990234375, 0.0799560546875, -0.0009341239929199219, -0.0204925537109375, -0.0301055908203125, -0.0086212158203125, -0.000598907470703125, 0.002902984619140625, -0.01058197021484375, -0.037261962890625, -0.018157958984375, -0.030487060546875, -0.010101318359375, 0.05499267578125, 0.0087890625, -0.0014238357543945312, -0.043243408203125, 0.049285888671875, 0.0219879150390625, 0.0023860931396484375, -0.0121002197265625, -0.006420135498046875, -0.001007080078125, -0.038116455078125, 0.0226898193359375, 0.305908203125, -0.01024627685546875, 0.00873565673828125, -0.011474609375, -0.014617919921875, -0.035003662109375, -0.0210113525390625, -0.00897979736328125, -0.0007367134094238281, -0.01904296875, 0.002414703369140625, 0.00638580322265625, 0.0200042724609375, -0.0165863037109375, -0.00011420249938964844, -0.043975830078125, -0.0018148422241210938, -0.04095458984375, 0.08599853515625, -0.0289306640625, 0.0202484130859375, 0.01360321044921875, -0.06036376953125, -0.0132293701171875, -0.017242431640625, -0.005672454833984375, 0.020477294921875, 0.00821685791015625, 0.01215362548828125, 0.0132598876953125, -0.0178985595703125, -0.02587890625, -0.0004062652587890625, 0.0288238525390625, 0.0160064697265625, 0.0709228515625, 0.0016727447509765625, -0.0179595947265625, 0.00919342041015625, 0.040374755859375, 0.035064697265625, 0.00423431396484375, 0.09295654296875, -0.0016031265258789062, 0.074951171875, 0.03704833984375, 0.0027790069580078125, 0.01369476318359375, 0.00434112548828125, 0.007465362548828125, -0.0101470947265625, -0.0034160614013671875, 0.006855010986328125, 0.00403594970703125, 0.06439208984375, -0.00199127197265625, -0.0301513671875, 0.01245880126953125, 0.01012420654296875, 0.160400390625, -0.01519775390625, 0.0328369140625, -0.05206298828125, -0.032196044921875, 0.040252685546875, 0.004119873046875, -0.024444580078125, 0.015380859375, 0.0190582275390625, 0.05352783203125, -0.006847381591796875, 0.0276641845703125, 0.003753662109375, 0.0155487060546875, 0.028350830078125, 0.0053253173828125, 0.03228759765625, 0.029022216796875, 0.015655517578125, 0.0249786376953125, 0.0084228515625, -0.01226806640625, 0.14111328125, 0.01018524169921875, 0.00591278076171875, -0.0457763671875, 0.04962158203125, 0.12249755859375, -0.005466461181640625, 0.01171112060546875, 0.02044677734375, 0.01390838623046875, -0.0267181396484375, -0.02044677734375, -0.0238037109375, 0.028106689453125, -0.01277923583984375, -0.027618408203125, 0.01200103759765625, 0.0130157470703125, 0.013763427734375, -0.024871826171875, -0.0312042236328125, 0.00943756103515625, 0.02471923828125, 0.0146331787109375, -0.021240234375, 0.036468505859375, -0.0030956268310546875, 0.030731201171875, 0.08294677734375, 0.01177978515625, 0.01273345947265625, -0.01251220703125, 0.052154541015625, 0.00972747802734375, -0.0025005340576171875, 0.003391265869140625, -0.0017185211181640625, 0.01068115234375, 0.00531768798828125, 0.007381439208984375, 0.00783538818359375, -0.00273895263671875, -0.0181732177734375, -0.01486968994140625, -0.0019550323486328125, -0.0022258758544921875, -0.039825439453125, -0.0389404296875, 0.01114654541015625, -0.045806884765625, 0.058319091796875, 0.05718994140625, -0.007061004638671875, 0.019744873046875, 0.06439208984375, 0.05206298828125, -0.0274200439453125, -0.03033447265625, -0.0061187744140625, -0.000896453857421875, 0.0012788772583007812, -0.0262908935546875, 0.013214111328125, -0.007450103759765625, 0.00411224365234375, -0.03826904296875, 0.0194244384765625, 0.0296630859375, 0.01436614990234375, 0.0172576904296875, 0.0308990478515625, -0.03997802734375, -0.0177154541015625, -0.01476287841796875, 0.042877197265625, -0.0145416259765625, 0.0102081298828125, -0.0076446533203125, -0.0002865791320800781, -0.0240020751953125, 0.042236328125, -0.01788330078125, 0.0394287109375, -0.0066375732421875, -0.0254974365234375, 0.01397705078125, -0.00928497314453125, -0.0063018798828125, 0.02197265625, -0.030670166015625, 0.0218048095703125, -0.038330078125, 0.077880859375, -0.027130126953125, 0.0160675048828125, -0.03179931640625, -0.01511383056640625, 0.042327880859375, -0.01180267333984375, -0.019927978515625, -0.01369476318359375, 0.02874755859375, -0.06884765625, -0.02935791015625, 0.0141754150390625, 0.023895263671875, -0.10198974609375, 0.00864410400390625, -0.0186309814453125, -0.00862884521484375, -0.03277587890625, -0.006511688232421875, 0.016387939453125, 0.02032470703125, -0.0555419921875, -0.00876617431640625, -0.0234375, -0.00803375244140625, 0.07476806640625, -0.0215301513671875, 0.017547607421875, -0.0014848709106445312, -0.0268707275390625, -0.050506591796875, -0.0223388671875, 0.0234832763671875, -0.0110321044921875, -0.04351806640625, -0.0157012939453125, -0.0249786376953125, 0.011260986328125, -0.0259246826171875, -0.06170654296875, 0.0250244140625, 0.0248260498046875, 0.0025634765625, -0.0199127197265625, 0.00760650634765625, 0.01151275634765625, -0.047698974609375, 0.06500244140625, -0.045623779296875, -0.0275726318359375, 0.027740478515625, -0.01224517822265625, -0.031341552734375, 0.01068115234375, -0.028778076171875, 0.024200439453125, -0.00658416748046875, -0.00147247314453125, -0.0027675628662109375, 0.01078033447265625, 0.0312042236328125, 0.025390625, -0.0032501220703125, -0.0120697021484375, 0.01123809814453125, -0.042266845703125, -0.039093017578125, 0.0157470703125, -0.017303466796875, 0.0645751953125, 0.0071868896484375, 0.03131103515625, -0.0300750732421875, -0.0275726318359375, 0.0312042236328125, 0.0130462646484375, 0.007015228271484375, -0.11285400390625, -0.0142669677734375, -0.0049896240234375, -0.0706787109375, 0.031524658203125, -0.0238037109375, 0.057769775390625, 0.00160980224609375, -0.004730224609375, 0.018646240234375, 0.0036907196044921875, -0.005031585693359375, -0.03253173828125, 0.046051025390625, 0.01561737060546875, -0.0251617431640625, -0.07537841796875, 0.00394439697265625, 0.048675537109375, -0.0161895751953125, 0.007434844970703125, -0.002956390380859375, 0.0484619140625, -0.0189208984375, 0.0032901763916015625, 0.10125732421875, -0.03790283203125, 0.02960205078125, -0.004825592041015625, -0.01499176025390625, 0.02642822265625, 0.0049591064453125, 0.02337646484375, -0.00927734375, -0.0494384765625, 0.0179595947265625, -0.027587890625, 0.0029354095458984375, -0.003871917724609375, 0.01244354248046875, -0.05169677734375, 0.04736328125, 0.01236724853515625, -0.00536346435546875, 0.032196044921875, -0.0333251953125, 0.00989532470703125, 0.020538330078125, 0.012298583984375, 0.003688812255859375, -0.001110076904296875, -0.029510498046875, -0.0252685546875, -0.051361083984375, 0.0139312744140625, -0.0262908935546875, 0.009124755859375, 0.045806884765625, 0.0013513565063476562, 0.06646728515625, -0.01922607421875, -0.0068206787109375, -0.031341552734375, -0.018218994140625, 0.0216827392578125, 0.002902984619140625, -0.0006070137023925781, -0.022735595703125, 0.0206451416015625, -0.0009202957153320312, -0.0069427490234375, -0.03411865234375, 0.01285552978515625, -0.0175628662109375, 0.019073486328125, 0.0191802978515625, -0.038909912109375, -0.0251312255859375, -0.03399658203125, -0.021087646484375, -0.02703857421875, -0.01374053955078125, -0.0291748046875, -0.017303466796875]') 
               AS rank
            FROM
               laion1m
            WHERE
               vec <#> '[-0.016510009765625, 0.016021728515625, 0.0310516357421875, 0.008148193359375, 0.0175628662109375, 0.002460479736328125, 0.0087127685546875, 0.0120086669921875, 0.06341552734375, 0.00652313232421875, 0.018218994140625, 0.02777099609375, 0.0518798828125, -0.006378173828125, -0.052764892578125, 0.0004036426544189453, 0.072509765625, -0.01102447509765625, 0.015167236328125, 0.013916015625, 0.10760498046875, 0.02191162109375, -0.004550933837890625, 0.02081298828125, -0.024261474609375, 0.028106689453125, 0.0662841796875, -0.006519317626953125, 0.0098114013671875, 0.042572021484375, -0.0148773193359375, 0.0205078125, -0.0238494873046875, -0.0006361007690429688, -0.01457977294921875, 0.04962158203125, 0.0012369155883789062, 0.0093994140625, 0.005023956298828125, 0.1170654296875, -0.02593994140625, 0.0313720703125, -0.0196685791015625, -0.0036525726318359375, -0.001354217529296875, -0.30126953125, -0.00830078125, 0.0128021240234375, 0.032196044921875, -0.0350341796875, 0.039825439453125, -0.00331878662109375, 0.01561737060546875, -0.00443267822265625, 0.0400390625, -0.031829833984375, -0.0164794921875, 0.026947021484375, 0.0177001953125, -0.0009918212890625, 0.043060302734375, 0.018218994140625, -0.00821685791015625, -0.07586669921875, 0.0199737548828125, -0.0172271728515625, 0.0450439453125, -0.047607421875, -0.048309326171875, 0.019775390625, -0.037841796875, -0.0166015625, -0.026092529296875, -0.02496337890625, -0.003635406494140625, -0.0126800537109375, -0.0731201171875, -0.052001953125, 0.006011962890625, -0.01519775390625, -0.00424957275390625, -0.034271240234375, -0.0220794677734375, 0.10284423828125, 0.00997161865234375, 0.0019359588623046875, -0.012298583984375, -0.0246429443359375, -0.09991455078125, -0.017181396484375, 0.007572174072265625, -0.0439453125, -0.50048828125, 0.07763671875, -0.00334930419921875, 0.0020694732666015625, 0.040008544921875, -0.0267486572265625, -0.055816650390625, -0.0299072265625, -0.01238250732421875, -0.0020847320556640625, -0.0262603759765625, 0.001373291015625, 0.0634765625, 0.000637054443359375, -0.1094970703125, -0.0265655517578125, 0.01611328125, -0.0166015625, 0.00965118408203125, -0.060577392578125, -0.0016536712646484375, 0.034393310546875, 0.0240631103515625, 0.007232666015625, 0.0184326171875, 0.0292816162109375, 0.0616455078125, 0.02447509765625, -0.0179290771484375, -0.053253173828125, 0.041778564453125, 0.0170745849609375, 0.0150146484375, 0.03155517578125, 0.0092010498046875, -0.0003840923309326172, -0.0009431838989257812, 0.05059814453125, 0.039215087890625, 0.01456451416015625, -0.0005426406860351562, 0.06475830078125, 0.00464630126953125, 0.0092926025390625, 0.0236663818359375, 0.007678985595703125, 0.00308990478515625, -0.0100555419921875, 1.430511474609375e-06, 0.00041866302490234375, -0.005756378173828125, -0.0015611648559570312, 0.000518798828125, 0.0352783203125, -0.02752685546875, 0.032501220703125, -0.0282135009765625, -0.00390625, 0.0362548828125, -0.0186614990234375, 0.0799560546875, -0.0009341239929199219, -0.0204925537109375, -0.0301055908203125, -0.0086212158203125, -0.000598907470703125, 0.002902984619140625, -0.01058197021484375, -0.037261962890625, -0.018157958984375, -0.030487060546875, -0.010101318359375, 0.05499267578125, 0.0087890625, -0.0014238357543945312, -0.043243408203125, 0.049285888671875, 0.0219879150390625, 0.0023860931396484375, -0.0121002197265625, -0.006420135498046875, -0.001007080078125, -0.038116455078125, 0.0226898193359375, 0.305908203125, -0.01024627685546875, 0.00873565673828125, -0.011474609375, -0.014617919921875, -0.035003662109375, -0.0210113525390625, -0.00897979736328125, -0.0007367134094238281, -0.01904296875, 0.002414703369140625, 0.00638580322265625, 0.0200042724609375, -0.0165863037109375, -0.00011420249938964844, -0.043975830078125, -0.0018148422241210938, -0.04095458984375, 0.08599853515625, -0.0289306640625, 0.0202484130859375, 0.01360321044921875, -0.06036376953125, -0.0132293701171875, -0.017242431640625, -0.005672454833984375, 0.020477294921875, 0.00821685791015625, 0.01215362548828125, 0.0132598876953125, -0.0178985595703125, -0.02587890625, -0.0004062652587890625, 0.0288238525390625, 0.0160064697265625, 0.0709228515625, 0.0016727447509765625, -0.0179595947265625, 0.00919342041015625, 0.040374755859375, 0.035064697265625, 0.00423431396484375, 0.09295654296875, -0.0016031265258789062, 0.074951171875, 0.03704833984375, 0.0027790069580078125, 0.01369476318359375, 0.00434112548828125, 0.007465362548828125, -0.0101470947265625, -0.0034160614013671875, 0.006855010986328125, 0.00403594970703125, 0.06439208984375, -0.00199127197265625, -0.0301513671875, 0.01245880126953125, 0.01012420654296875, 0.160400390625, -0.01519775390625, 0.0328369140625, -0.05206298828125, -0.032196044921875, 0.040252685546875, 0.004119873046875, -0.024444580078125, 0.015380859375, 0.0190582275390625, 0.05352783203125, -0.006847381591796875, 0.0276641845703125, 0.003753662109375, 0.0155487060546875, 0.028350830078125, 0.0053253173828125, 0.03228759765625, 0.029022216796875, 0.015655517578125, 0.0249786376953125, 0.0084228515625, -0.01226806640625, 0.14111328125, 0.01018524169921875, 0.00591278076171875, -0.0457763671875, 0.04962158203125, 0.12249755859375, -0.005466461181640625, 0.01171112060546875, 0.02044677734375, 0.01390838623046875, -0.0267181396484375, -0.02044677734375, -0.0238037109375, 0.028106689453125, -0.01277923583984375, -0.027618408203125, 0.01200103759765625, 0.0130157470703125, 0.013763427734375, -0.024871826171875, -0.0312042236328125, 0.00943756103515625, 0.02471923828125, 0.0146331787109375, -0.021240234375, 0.036468505859375, -0.0030956268310546875, 0.030731201171875, 0.08294677734375, 0.01177978515625, 0.01273345947265625, -0.01251220703125, 0.052154541015625, 0.00972747802734375, -0.0025005340576171875, 0.003391265869140625, -0.0017185211181640625, 0.01068115234375, 0.00531768798828125, 0.007381439208984375, 0.00783538818359375, -0.00273895263671875, -0.0181732177734375, -0.01486968994140625, -0.0019550323486328125, -0.0022258758544921875, -0.039825439453125, -0.0389404296875, 0.01114654541015625, -0.045806884765625, 0.058319091796875, 0.05718994140625, -0.007061004638671875, 0.019744873046875, 0.06439208984375, 0.05206298828125, -0.0274200439453125, -0.03033447265625, -0.0061187744140625, -0.000896453857421875, 0.0012788772583007812, -0.0262908935546875, 0.013214111328125, -0.007450103759765625, 0.00411224365234375, -0.03826904296875, 0.0194244384765625, 0.0296630859375, 0.01436614990234375, 0.0172576904296875, 0.0308990478515625, -0.03997802734375, -0.0177154541015625, -0.01476287841796875, 0.042877197265625, -0.0145416259765625, 0.0102081298828125, -0.0076446533203125, -0.0002865791320800781, -0.0240020751953125, 0.042236328125, -0.01788330078125, 0.0394287109375, -0.0066375732421875, -0.0254974365234375, 0.01397705078125, -0.00928497314453125, -0.0063018798828125, 0.02197265625, -0.030670166015625, 0.0218048095703125, -0.038330078125, 0.077880859375, -0.027130126953125, 0.0160675048828125, -0.03179931640625, -0.01511383056640625, 0.042327880859375, -0.01180267333984375, -0.019927978515625, -0.01369476318359375, 0.02874755859375, -0.06884765625, -0.02935791015625, 0.0141754150390625, 0.023895263671875, -0.10198974609375, 0.00864410400390625, -0.0186309814453125, -0.00862884521484375, -0.03277587890625, -0.006511688232421875, 0.016387939453125, 0.02032470703125, -0.0555419921875, -0.00876617431640625, -0.0234375, -0.00803375244140625, 0.07476806640625, -0.0215301513671875, 0.017547607421875, -0.0014848709106445312, -0.0268707275390625, -0.050506591796875, -0.0223388671875, 0.0234832763671875, -0.0110321044921875, -0.04351806640625, -0.0157012939453125, -0.0249786376953125, 0.011260986328125, -0.0259246826171875, -0.06170654296875, 0.0250244140625, 0.0248260498046875, 0.0025634765625, -0.0199127197265625, 0.00760650634765625, 0.01151275634765625, -0.047698974609375, 0.06500244140625, -0.045623779296875, -0.0275726318359375, 0.027740478515625, -0.01224517822265625, -0.031341552734375, 0.01068115234375, -0.028778076171875, 0.024200439453125, -0.00658416748046875, -0.00147247314453125, -0.0027675628662109375, 0.01078033447265625, 0.0312042236328125, 0.025390625, -0.0032501220703125, -0.0120697021484375, 0.01123809814453125, -0.042266845703125, -0.039093017578125, 0.0157470703125, -0.017303466796875, 0.0645751953125, 0.0071868896484375, 0.03131103515625, -0.0300750732421875, -0.0275726318359375, 0.0312042236328125, 0.0130462646484375, 0.007015228271484375, -0.11285400390625, -0.0142669677734375, -0.0049896240234375, -0.0706787109375, 0.031524658203125, -0.0238037109375, 0.057769775390625, 0.00160980224609375, -0.004730224609375, 0.018646240234375, 0.0036907196044921875, -0.005031585693359375, -0.03253173828125, 0.046051025390625, 0.01561737060546875, -0.0251617431640625, -0.07537841796875, 0.00394439697265625, 0.048675537109375, -0.0161895751953125, 0.007434844970703125, -0.002956390380859375, 0.0484619140625, -0.0189208984375, 0.0032901763916015625, 0.10125732421875, -0.03790283203125, 0.02960205078125, -0.004825592041015625, -0.01499176025390625, 0.02642822265625, 0.0049591064453125, 0.02337646484375, -0.00927734375, -0.0494384765625, 0.0179595947265625, -0.027587890625, 0.0029354095458984375, -0.003871917724609375, 0.01244354248046875, -0.05169677734375, 0.04736328125, 0.01236724853515625, -0.00536346435546875, 0.032196044921875, -0.0333251953125, 0.00989532470703125, 0.020538330078125, 0.012298583984375, 0.003688812255859375, -0.001110076904296875, -0.029510498046875, -0.0252685546875, -0.051361083984375, 0.0139312744140625, -0.0262908935546875, 0.009124755859375, 0.045806884765625, 0.0013513565063476562, 0.06646728515625, -0.01922607421875, -0.0068206787109375, -0.031341552734375, -0.018218994140625, 0.0216827392578125, 0.002902984619140625, -0.0006070137023925781, -0.022735595703125, 0.0206451416015625, -0.0009202957153320312, -0.0069427490234375, -0.03411865234375, 0.01285552978515625, -0.0175628662109375, 0.019073486328125, 0.0191802978515625, -0.038909912109375, -0.0251312255859375, -0.03399658203125, -0.021087646484375, -0.02703857421875, -0.01374053955078125, -0.0291748046875, -0.017303466796875]' < -0.81
         ) AS ranked
      WHERE
         ranked.rank <= 10;