# Agente Autónomo

**Un agente de IA local diseñado para razonar, utilizar herramientas y ejecutar tareas de forma autónoma.**

Agente Autónomo es un proyecto personal en desarrollo cuyo objetivo es construir un asistente capaz de ir más allá de una conversación: analizar una petición, planificar los pasos necesarios, utilizar herramientas, comprobar resultados y continuar trabajando hasta completar una tarea.

El proyecto nace de una idea sencilla: quiero entender cómo construir un agente de verdad, desde su arquitectura interna hasta la interacción con los modelos de lenguaje, sin depender por completo de soluciones cerradas.

## Objetivo

Construir un agente modular que pueda interpretar instrucciones, tomar decisiones durante la ejecución y trabajar con herramientas reales. La intención es evolucionar progresivamente hacia un sistema capaz de abordar tareas complejas, mantener el contexto y adaptarse a distintos modelos de IA.

## ¿Cómo funciona?

El proyecto separa la conversación directa del flujo de ejecución autónoma:

- **Chat:** interacción directa con el modelo de lenguaje.
- **Agente:** flujo de trabajo que interpreta la petición, planifica, selecciona herramientas, ejecuta acciones y verifica los resultados.
- **Análisis:** modo orientado al análisis de información y código.

La arquitectura se organiza en componentes para separar responsabilidades y facilitar su evolución.

## Tecnologías

- **Python** y programación asíncrona.
- **Ollama** para la comunicación con modelos de lenguaje locales.
- **Pydantic** para la validación y estructuración de datos.
- **ChromaDB** para funcionalidades de almacenamiento y recuperación vectorial.
- **AsyncIO y HTTPX** para operaciones asíncronas y comunicación HTTP.

## Estado del proyecto

Agente Autónomo es un proyecto experimental en evolución. El objetivo actual es consolidar el flujo de ejecución, mejorar la integración de herramientas y separar la lógica del agente del proveedor de modelos para poder incorporar alternativas en el futuro.

Las capacidades disponibles y su grado de estabilidad pueden cambiar a medida que avance el desarrollo.

## Filosofía

Este proyecto es también una forma de aprender construyendo: experimentar con arquitecturas de agentes, estudiar sus limitaciones y convertir esas ideas en software funcional.

No se trata únicamente de conectar un modelo con unas herramientas, sino de entender cómo coordinar razonamiento, acciones, contexto y verificación dentro de un sistema mantenible.

---

**Estado:** En desarrollo activo.

**Autor:** [Renzi](https://renzi.dev)
