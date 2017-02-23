module Mahony

  importall Util

  th = 0.45
  MAX_RATE = 5
  MIN_RATE = 1

  function mahony(predictRate, rate)
    return abs(rate - predictRate) / (MAX_RATE - MIN_RATE) > th
  end

  function correct(k, usersTraining, simVector)
    println("Corrigindo Ruido Mahony")
    #percorre todos os usuários de treino

    correctQuant = 0

    for u in eachindex(usersTraining[:,1])
      #Itens que vamos avaliar
      itens = find(x -> x > 0, usersTraining[u,:])

      for i in itens
        prediction = Util.predict(u, i, k, simVector, usersTraining)
        if isnan(prediction)
          continue
        end
        if mahony(prediction, usersTraining[u, i])
          usersTraining[u, i] = round(prediction)
          correctQuant += 1
        end
      end
    end
    println("Foram corrigidos $correctQuant rate")
    println("Corrigindo Ruido Mahony terminado")
  end

end


treino normal 
verifica se cada nota está do mahony, se não subtuo a original pela previsão
feito isso na mtriz da previão inteira

treina de novo com a matriz corrigida