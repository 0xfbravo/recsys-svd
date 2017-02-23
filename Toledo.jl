
module Toledo

  importall Util

  th = 0.2
  MAX_RATE = 5
  MIN_RATE = 1

  critical = "CRITICAL"
  average = "AVERAGE"
  benevolent = "BENEVOLENT"
  variable = "VARIABLE"

  weakly = "WEAKLY"
  averagely = "AVERAGELY"
  strongly = "strongly"
  variably = "VARIABLY"

  k = ku = ki =  MIN_RATE + round(1/3 * (MAX_RATE - MIN_RATE))
  v = vu = vi =  MAX_RATE - round(1/3 * (MAX_RATE - MIN_RATE))

  function detectNoise(usersTraining)

    Wu = Dict()
    Wi = Dict()
    Au = Dict()
    Ai = Dict()
    Su = Dict()
    Si = Dict()
    noise = []

    userClass = Dict()
    itemClass = Dict()

    #percorre todos os usuários de treino
    for u in eachindex(usersTraining[:,1])
      #Itens que vamos avaliar
      itens = find(x -> x > 0, usersTraining[u,:])

      Wu[u] = []
      Au[u] = []
      Su[u] = []

      for i in itens

        if !haskey(Wi, i)
          Wi[i] = []
        end

        if !haskey(Ai, i)
          Ai[i] = []
        end

        if !haskey(Si, i)
          Si[i] = []
        end

        if usersTraining[u, i] < ku
          append!(Wu[u], usersTraining[u, i])
        elseif usersTraining[u, i] >= ku && usersTraining[u, i] < vu
          append!(Au[u], usersTraining[u, i])
        else
          append!(Su[u], usersTraining[u, i])
        end

        if usersTraining[u, i] < ki
          append!(Wi[i], usersTraining[u, i])
        elseif usersTraining[u, i] >= ki && usersTraining[u, i] < vi
          append!(Ai[i], usersTraining[u, i])
        else
          append!(Si[i], usersTraining[u, i])
        end

      end
    end

    #percorre todos os usuários de treino
    for u in eachindex(usersTraining[:,1])
      if (size(Wu[u])[1] >= size(Au[u])[1] + size(Su[u])[1])
        userClass[u] = critical
      elseif (size(Au[u])[1] >= size(Wu[u])[1] + size(Su[u])[1])
        userClass[u] = average
      elseif (size(Su[u])[1] >= size(Au[u])[1] + size(Wu[u])[1])
        userClass[u] = benevolent
      else
        userClass[u] = variable
      end
    end

    #percorre todos os itens de treino
    for i in eachindex(usersTraining[1,:])
      if (!haskey(Wi, i))
        continue
      end
      if (size(Wi[i])[1] >= size(Ai[i])[1] + size(Si[i])[1])
        itemClass[i] = weakly
      elseif (size(Ai[i])[1] >= size(Wi[i])[1] + size(Si[i])[1])
        itemClass[i] = averagely
      elseif (size(Si[i])[1] >= size(Ai[i])[1] + size(Wi[i])[1])
        itemClass[i] = strongly
      else
        itemClass[i] = variably
      end
    end

    #percorre todos os usuários de treino
    for u in eachindex(usersTraining[:,1])
      #Itens que vamos avaliar
      itens = find(x -> x > 0, usersTraining[u,:])

      for i in itens
        if userClass[u] == critical && itemClass[i] == weakly && usersTraining[u, i] > k
          append!(noise, [(u, i)])
        end

        if userClass[u] == average && itemClass[i] == averagely && (usersTraining[u, i] < k || usersTraining[u, i] >= v)
          append!(noise, [(u, i)])
        end

        if userClass[u] == benevolent && itemClass[i] == strongly && usersTraining[u, i] < v
          append!(noise, [(u, i)])
        end
      end
    end
    return noise
  end

  function correct(k, usersTraining, simVector)
    possibleNoise = detectNoise(usersTraining)
    correctQuant = 0
    for (u, i) in possibleNoise
      prediction = Util.predict(u, i, k, simVector, usersTraining)
      if (isnan(prediction))
        continue
      end
      if (abs(prediction - usersTraining[u, i]) > th)
        usersTraining[u, i] = round(prediction)
        correctQuant += 1
      end
    end
    println("Foram corrigidos $correctQuant rate")
  end
end
