# BikeBalance

BikeBalance is a project I have started working on during my fellowship at Insight Data Science. 

## Problem statement
Users of a bike-share service expect that that each bike station in the network will always have a bike available when they need one and also an empty dock to drop of a bike. This is acheived by station "rebalancing": removing the bikes from the stations that do not have enough docks and bringing the bikes to the stations that do not have enough bikes. Bike balancing, often done ad-hoc and by eye, faces challenges due to traffic and weather and becomes unfeasible as the bike share network continues to grow. I developed a tool that predicts day-to-day bike demand across the network to plan efficient and timely station re-balancing.

## Solution
I used data about the ride frequency in the past three years together with historic weather data to predict bike demand (defined as the number of bikes/per hour that leave a station on a given day) for each station in the network. I created a web-app that graphically shows predicted inflow and outflow of bikes for a select station on a given day and  displays a warning if either the station is likely to get out-of-balance (run out of docks or bikes). The tool allows the bike-share service to assess which stations are likely to require rebalancing and plan accordingly.

## Project presentation link
https://tinyurl.com/bikebalance

## Web-app link
http://bikebalance.info/
