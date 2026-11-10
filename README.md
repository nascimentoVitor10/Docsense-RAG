# Docsense RAG - Enxergando o Brasil

## O que é?

## Arquitetura do projeto

## Tecnologias utilizadas

## Fontes de dados

## Aprendizado

* pasta CORE/ : em projetos python costuma reunir componentes que são utilizados em diferentes
partes da aplicação. Nela podemos colocar *configurações globais*, *variáveis de ambiente*, 
além de outros recursos transversais.

Exemplo:
    - a API precisa saber o nome do ambiente e a versão da aplicação
    - o módulo de embbedings precisa saber as configurações do modelo
    - o módulo de recuperação precisa conhecer os parâmetros de busca

A ideia central é tornar o núcleo um agregador das diferentes configurações
para que cada configuração não fique em um lugar diferente.


- Por que o código da API não deve conter toda a lógica do RAG?
> a lógica de RAG deve estar fracamente acoplada da API para que, ao receber manutenção
a API continue disponível para os usuários mesmo com as alterações na lógica de RAG.


- Qual é a diferença entre uma configuração e uma regra de negócio?
> uma regra de negócio diz respeito a como um processo irá ser implementado, enquanto
uma configuração expõe um parêmetro que alimenta um processo.


- Por que testar uma rota HTTP não substitui os testes unitários?
> 


- Como sabemos que uma mudança não quebrou funcionalidades que já estavam funcionando?
> por meio de aplicações de testes automatizados (?)


### Etapa 2 : Construção do Corpus

* importante definir como um documento será identificado e descrito dentro do sistema :
é importante para manter a rastreabilidade, a identificação de duplicatas e a capacidade
de reprocessar o corpus

* Manifest.jsonl (catálogo explícito de documentos): arquivo manifesto é um arquivo que 
descreve um conjunto de recursos que serão utilizados pelo sistema. Ele reúne metadados 
a respeito dos recursos, tais como a localidade onde estes serão baixados.

Exemplo:
    
    {"document_id":"edital-001","title":"Aquisição de computadores","source_url":"https://exemplo.gov.br/edital-001.pdf","document_type":"edital","organization":"Secretaria de Tecnologia"}
    {"document_id":"tr-002","title":"Especificação dos equipamentos","source_url":"https://exemplo.gov.br/tr-002.pdf","document_type":"termo_referencia","organization":"Secretaria de Tecnologia"}

* Foi construído a parte responsável por validar a estrutura do documento presente no manifesto
e outra para acessar e ler o manifesto.

### Etapa 3 : Ingestão dos Dados

* antes de tudo, precisamos definir as responsabilidades e os contratos antes da implementação.


* Dataclasses : é um decorador nativo que implementa 3 métodos mágicos (__ini__, __repr__,__eq__) de forma padrão, além de poder
implementar métodos de comparação. Esse decorador basicamente viabiliza a redução de códigos implementados.

* Biblioteca *tempfile* : cria arquivos e diretórios temporários de forma segura. Esse módulo é útil quando é preciso armazenar dados temporariamente durante a execução do programa e ter a garantia da sua exclusão de forma segura.

* A pasta *scripts* : tem a finalidade de reunir os comandos operacionais usados para executar tarefas do código que implementa as regras de negócio do sistema.

* Foi adicionada uma etapa de verificação se o arquivo baixado é mesmo um PDF. Além disso, foi calculado um hash para cada arquivo (hash SHA-256)

### Etapa 4 : Extração de Textos dos PDF's

* antes de tudo, para poder modelar as classes e funções, é preciso pensar no domínio do problema e suas invariantes que precisam ser preservadas.

* Foi criado um Parser , no qual duas classes foram definidas : ParsedPage e ParsedDocument.

* Será definida uma classe *Chunk* para poder manipular os trechos de texto.

* um chunk pode começar em uma página e terminar em outra, logo, é necessário manter a estrutura que está no documento para não perder contexto.

* o tamanho do chunk deve ser configurável, dado que tokens não são universais. Um trecho de texto pode ser representado por diferentes quantidade de tokens de acordo com o tokenizador e o modelo utilizado.

* *Sobreposição* : ajuda quando uma definição fica na fronteira entre dois trechos.

* Além do modelo *Chunk*, será implementado o *Chunker*,responsável por transformar o texto extraído em unidades que podem ser recuperadas e indexadas pelo RAG.

* necessário definir uma estratégia de *Segmentação* com base no tipo de documentos trabalhados.

* *Chunking* : é o processo de divir um documento extenso em unidades menores de texto que podem ser indexadas, recuperadas e fornecidas a um modelo de linguagem

    - a divisão de texto influencia diretamente na recuperação do texto

    - objetivo : encontrar um equilíbrio entre CONTEXTO SUFICIENTE, TAMANHO CONTROLADO e PRECISÃO NA RECUPERAÇÃO

    tipos de chuncking:

        * chunking por tamanho fixo
        * chunking estrutural
        * chunking com overlap
        * chunking híbrido

### Etapa 5 : Construção de Chunks

* parte mais chata e complicada até o momento

### Etapa 6 : Embeddings