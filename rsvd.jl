using MLBase

# Config
PATH = "/Users/insidemybrain/recsys-svd/"

# readFile - MovieLens 100k (u.data)
function readFile(dir)
  println("Reading data...")
  file = open(dir,"r")
  content = readdlm(file)
  close(file)
  return content
end

# Create a Rates Matrix (UsersxItens) - @GSegobia
function rates_matrix(file_content)
  usersNo = convert(Int64,maximum(file_content[:,1]))
  itemsNo = convert(Int64, maximum(file_content[:,2]))
  complete_users_rates = zeros(usersNo, itemsNo)
  for i in 1:usersNo
    user = find(x->(x == i), file_content[:, 1])
    for j in 1:length(user)
      complete_users_rates[i, convert(Int64, file_content[user[j], 2])] = file_content[user[j], 3]
    end
  end
  return complete_users_rates
end

# RSVD Training - @filipebraida @insidemybrain
function rsvd_training(matrix, lrate = .0003, λ = .0013, Δ = .00032)
  println("Running RSVD Training...")
  (U,S,I) = svd(matrix)

  # 10 Most important Values
  U = U[:,1:10]
  I = I[:,1:10]

  # Users and Items Indexes where Rate != 0 (Vectors)
  (users, items) = ind2sub(size(matrix), find(r->r!=0, matrix))
  usersHeight = size(users)[1]

  # Error Vector
  baseError = zeros(usersHeight,1)
  iterationError = zeros(usersHeight,1)

  # Training
  keepTrying = true
  iteration = 1
  while(keepTrying)
    for i=1:usersHeight
      baseError[i,1] = matrix[users[i,1],items[i,1]] - (U[users[i,1]]' * I[items[i,1]])[1]

      U[users[i,1]] += lrate * (baseError[i,1] * I[items[i,1]] - λ * U[users[i,1]])
      I[items[i,1]] += lrate * (baseError[i,1] * U[users[i,1]] - λ * I[items[i,1]])
    end
    iterationError[iteration] = mean(abs(baseError))
    keepTrying = !(iteration > 1 && (mean(baseError) < Δ || iterationError[iteration] > iterationError[iteration-1]))
    iteration += 1
  end
  println("Finished RSVD Training with ",iteration," iterations.")
  writedlm(string(PATH,"rsvd_trained_users.data"), U)
  writedlm(string(PATH,"rsvd_trained_items.data"), I)
  writedlm(string(PATH,"rsvd_trained_errors.data"), baseError)
  return (U,I)
end

# RSVD Prediction - @insidemybrain
function rsvd_prediction(matrix, U, I)
  println("Running RSVD Prediction...")

  # Users and Items Indexes where Rate != 0 (Vectors)
  (users, items) = ind2sub(size(matrix), find(r->r!=0, matrix))
  usersHeight = size(users)[1]

  prediction = zeros(usersHeight,3)
  for i=1:usersHeight
    prediction[i,1] = users[i,1]
    prediction[i,2] = items[i,1]
    prediction[i,3] = abs((U[users[i,1]]' * I[items[i,1]])[1])
  end
  writedlm(string(PATH,"rsvd_prediction.data"), prediction)
  return prediction
end

# Paulo
function r(matrix,slice)
  newMatrix = copy(matrix)
  k = setdiff(shuffle(1:100000),slice)
  newMatrix[k,3] .= -1
  writedlm("novaMatriz",newMatrix)
  return newMatrix
end

# MAE
function mae(prediction,original)
  println("Running RSVD Training...")
  maeResults=[]
  p = sortrows(prediction, by=x->(x[2],x[1]))
  o = sortrows(original, by=x->(x[2],x[1]))

  globalMean = mean(original[:,3])
  for item=1:maximum(original[:,2]) # For Each Item
    originalRatesOfItem = find(r->r==item,original[:,2])
    predictedRatesOfItem = find(r->r==item,prediction[:,2])
    meanActualItem = mean(abs(prediction[predictedRatesOfItem,3] - original[originalRatesOfItem,3]))
    push!(maeResults,meanActualItem)
    println(meanActualItem)
  end

  writedlm(string(PATH,"sort/sorted_prediction.data"),p)
  writedlm(string(PATH,"sort/sorted_original.data"),o)
  writedlm(string(PATH,"rsvd_mae_result.data"),maeResults)
  return maeResults
end

# Change Rate
function change_rate(rate)
  if (rate == 1) return 5
  elseif (rate == 2) return 4
  elseif (rate == 3) return 2
  elseif (rate == 4) return 3
  elseif (rate == 5) return 1
  else return 0
  end
end

# Noise
function add_noise(original, percentage = 1)
  base = find(r->r, shuffle(1:100000) .> 100000 - percentage * 1000)
  for i in 1:size(base)[1]
    original[base[i],3] = change_rate(original[i,3])
  end
  return (original,base)
end

# Mahony's Algorithm
function mahony(prediction, rate, min = 1, max = 5, th = .5)
  return find(r-> r > th, abs(rate - prediction) ./ (max - min))
end

function mahony_correction(prediction,original,base)
  println("Running Mahony Correction...")
  o = sortrows(original, by=x->(x[2],x[1]))
  mah = mahony(prediction[:,3],o[:,3])
  matrixBase = zeros(Bool,100000)
  matrixBase[base] = true
  matrixMah = zeros(Bool,100000)
  matrixMah[mah] = true
  return (length(mah),roc(matrixBase,matrixMah))
end

# Running...
training = find(r->r, shuffle(1:100000) .> 20000) # 80k
test = setdiff(1:100000,training) # 20k

originalMatrix = readFile(string(PATH,"ml-100k/u.data"))
(originalMatrix, base) = add_noise(originalMatrix,1.7);
matrix20 = r(originalMatrix,test)
ratesMatrixTraining = rates_matrix(originalMatrix[training,:])
ratesMatrixTest = rates_matrix(matrix20)
(U,I) = rsvd_training(ratesMatrixTraining)
prediction = rsvd_prediction(ratesMatrixTest, U, I)
(mahonyCorrections,rocBalboa) = mahony_correction(prediction,originalMatrix,base)
println("Mahony's Corrections: ",mahonyCorrections," itens")
println("Precision: ",precision(rocBalboa)," Recall:",recall(rocBalboa)," F1:",f1score(rocBalboa))
#@time maeResults = mae(prediction,originalMatrix)
#println(mean(maeResults))
