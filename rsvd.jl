# Definitions
USERS_NUMBER = 943
ITENS_NUMBER = 1682

# readFile - MovieLens 100k (u.data)
function readFile(dir)
  println("Reading data...")
  file = open(dir)
  content = readdlm(file)
  close(file)
  return content
end

# Create a Rates Matrix (UsersxItens) - @GSegobia
function rates_matrix(file_content)
  complete_users_rates = zeros(USERS_NUMBER, ITENS_NUMBER)
  for i in 1:USERS_NUMBER
    user = find(x->(x == i), file_content[:, 1])
    for j in 1:length(user)
      complete_users_rates[i, convert(Int64, file_content[user[j], 2])] = file_content[user[j], 3]
    end
  end
  return complete_users_rates
end

# RSVD Training - @filipebraida @insidemybrain
function rsvd_training(matrix, maxIterations = 120, lrate = .001, λ = .02, Δ = .03)
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
  iterationError = zeros(maxIterations,1)

  # Training
  keepTrying = true
  iteration = 1
  while(keepTrying)
    for i=1:usersHeight
      baseError[i,1] = matrix[users[i],items[i]] - (U[users[i,1]]' * I[items[i,1]])[1]

      U[users[i],:] += lrate * (baseError[i,1] * I[items[i],:] - λ * U[users[i],:])
      I[items[i],:] += lrate * (baseError[i,1] * U[users[i],:] - λ * I[items[i],:])
    end
    iterationError[iteration] = mean(abs(baseError))
    keepTrying = !(iteration > 1 && (mean(baseError) < Δ || iterationError[iteration] > iterationError[iteration-1]))
    iteration += 1
  end
  println("Finished RSVD Training with ",iteration," iterations.")
  writedlm("rsvd_trained_users.data", U)
  writedlm("rsvd_trained_items.data", I)
  writedlm("rsvd_trained_errors.data", baseError)
  return (U,I)
end

# RSVD Prediction - @insidemybrain
function rsvd_prediction(matrix, U, I)
  println("Running RSVD Prediction...")
  (users, items) = ind2sub(size(matrix), find(r->r!=0, matrix))
  usersHeight = size(users)[1]
  prediction = zeros(usersHeight,1)
  for i=1:usersHeight # Users Count
    #todo
  end
  writedlm("rsvd_prediction.data", prediction)
end

# Running...
@time ratesMatrix = rates_matrix(readFile("ml-100k/u.data"))
@time (U,I) = rsvd_training(ratesMatrix)
@time rsvd_prediction(ratesMatrix, U, I)
