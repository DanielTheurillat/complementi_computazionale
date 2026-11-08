# Complementi di Fisica Computazionale: Il modello XY 2D

L'obbiettivo di questo complemento è la simulazione numerica del modello XY in due dimensioni 
tramite l'algoritmo Metropolis-Hastings.

In particolare siamo interessati alle misure di energia, magnetizzazione, calore specifico e 
suscettività magnetica. Queste misure vengono effettuate sulle configurazioni generate
dalla MCMC (Monte Carlo Markov Chain).

Sono state ripetute molte simulazioni variando sia le dimensioni del reticolo che la temperatura
del sistema.

## Termalizzazione

Prima di poter effettuare le misure è necessario assicurarsi che la configurazione abbia raggiunto
uno stato di equilibrio, dipendente dalla temperatura scelta. Questo può richiedere un numero di
steps diverso in base alla temperatura e alle dimensioni del reticolo. Per steps si intende un
numero di aggiornamenti locali di Metropolis pari al numero di siti totali, ovvero `L^2` per un
reticolo quadrato, con `L` la larghezza. <br>
Un'altro fattore che influenza il tempo di termalizzazione è il tipo di configurazione di partenza,
infatti sono possibili due configurazioni iniziali: *hot start* e *cold start*. Per *hot start* si
intende una configurazione iniziale in cui tutti gli spin sono scelti in maniera randomica. Invece
per *cold start* si intende invece una configurazione iniziale in cui tutti gli spin sono allineati
ad una data direzione.<br>
Qui di seguito sono riportati vari grafici dell'energia e della magnetizzazione.

### *Cold Start*

<img src=assets/images/en_magn_cold_T02.png width=50%>