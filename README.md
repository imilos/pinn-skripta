# Неуронске мреже подржане физичким законима - Практикум

Ово је званични репозиторијум материјала *Неуронске мреже подржане физичким законима - Практикум*. Предмет овог уџбеничког материјала су *Physics Informed Neural Networks (PINN)*, нови концепт моделовања који користи физичке законе исказане у облику обичних или парцијалних диференцијалних једначина као регуларизациони агент при тренингу дубоких неуронских мрежа. На овај начин се постиже боља тачност у односу на "обичне" дубоке мреже и отварају многе нове могућности инверзног моделовања (идентификације параметара), асимилације мерења, итд.

Практикум је првенствено намењен студентима мастер и докторских студија Рачунарских наука на [Природно-математичком факултету](http://www.pmf.kg.ac.rs) на [Универзитету у Крагујевцу](http://www.kg.ac.rs), али га могу користити и сви други којима је ова материја интересантна. 

## Издања
Практикум је доступан и у ћириличном и у латиничном издању. 

### Ћирилично издање
* HTML издање практикума доступно је на: :link: https://imi.pmf.kg.ac.rs/~milos/pinn/
* PDF издање практикума се може преузети са: :link: https://scidar.kg.ac.rs/handle/123456789/18602
* EPUB издање практикума се може преузети са: :link: https://imi.pmf.kg.ac.rs/~milos/pinn/pinn-skripta.epub

### Латинично издање
* HTML издање практикума доступно је на: :link: https://imi.pmf.kg.ac.rs/~milos/pinn-lat/
* PDF издање практикума се може преузети са: :link: https://scidar.kg.ac.rs/handle/123456789/18602
* EPUB издање практикума се може преузети са: :link: https://imi.pmf.kg.ac.rs/~milos/pinn-lat/pinn-skripta.epub

## Изворни код

* Комплетни примери обрађени у практикуму могу се наћи на :link: https://github.com/imilos/pinn-skripta/tree/main/primeri
* Практикум је писан у [Sphinx](http://www.sphinx-doc.org) генератору документације и читав изворни код је доступан у овом репозиторијуму. 

Под условом да је инсталиран [Sphinx](http://www.sphinx-doc.org), командe за изградњу HTML, Latex и EPUB издања су следеће (респективно):

    sphinx-build -b html . html/
    sphinx-build -b latex . tex/; ./latex-custom.sh; cd tex; make; cd ..
    sphinx-build -b epub -D extensions=sphinx.ext.imgmath,sphinxcontrib.bibtex -D imgmath_embed=True . epub/

Изворни фајлови су ћирилични, а латинична верзија се добиja аутоматским пресловљавањем помоћу скрипте `preslovi.sh`. 

## Питања и предлози

Уколико имате било каквих питања, можете их поставити у [Issues](https://github.com/imilos/pinn-skripta/issues) одељку.

## Ауторска права [![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

Нека права задржана. Ово дело обjављено jе под условима [Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa]).

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0
[cc-by-sa-image]: https://mirrors.creativecommons.org/presskit/buttons/88x31/svg/by-sa.svg
[cc-by-sa-shield]: https://mirrors.creativecommons.org/presskit/buttons/80x15/svg/by-sa.svg
