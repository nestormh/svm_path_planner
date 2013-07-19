/**
 * 
 */
package sibtra.agentes;

import jade.core.Agent;
import jade.domain.DFService;
import jade.domain.FIPAException;
import jade.domain.FIPANames;
import jade.domain.FIPAAgentManagement.DFAgentDescription;
import jade.domain.FIPAAgentManagement.FailureException;
import jade.domain.FIPAAgentManagement.NotUnderstoodException;
import jade.domain.FIPAAgentManagement.RefuseException;
import jade.domain.FIPAAgentManagement.ServiceDescription;
import jade.lang.acl.ACLMessage;
import jade.lang.acl.MessageTemplate;
import jade.proto.SimpleAchieveREResponder;

import java.util.StringTokenizer;
import java.util.regex.Pattern;

import sibtra.lidar.BarridoAngular;
import sibtra.lms.LMSException;
import sibtra.lms.ManejaLMS221;
import sibtra.lms.ManejaTelegramasIO;
import sibtra.lms.ZonaLMS;


/**
 * Agente que estará en contacto con el LSM a través de la serial.
 * Hará la configuración inicial y atenderá las peticiones de información de otros agentes.
 * 
 * No hay control de tiempo en las respuestas seriales. Se puede quedar bloquedo esperando la respuesta
 * desde el LMS.
 * 
 * @author alberto
 *
 */
public class AgenteLMS extends Agent {

	
	/**
	 * Bandera para que los comportamientos sepan si otro está usando el LMS
	 */
	protected boolean UsandoLMS=true;
	
	/**
	 * Para activar o desactivar los logs
	 */
	private boolean logs;

	private ManejaLMS221 manLMS;
	
	/**
	 * Metodo para emitir los mensajes de Log
	 */
	protected void agentLog(String m) {
		if(logs)
			System.out.println(getAID().getName()+
					": "+System.currentTimeMillis()+": "+m);
	}
	
	
	protected void setup() {
		agentLog("Se arranca el agenteLMS ");
		UsandoLMS=true;	//LMS ocupado
			
		//Puerto serie por defecto
		String puertoSerie="/dev/ttyS0";
		logs=true;
		boolean usaRXTX=false;
		//analizamos argumentos
		Object[] args = getArguments();
		if (args!=null) 
			for(int i=0; i<args.length; i++) {
				String aa=args[i].toString();
				agentLog("Argumento "+ i + " es '"+aa+"'");
				if(aa.startsWith("puerto:") && aa.length()>"puerto:".length() ) {
					puertoSerie=aa.substring(aa.indexOf(":")+1);
				}
				if(aa.startsWith("logs:") && aa.length()>"logs:".length() ) {
					logs=aa.substring(aa.indexOf(":")+1).equalsIgnoreCase("on");
				}
				if(aa.startsWith("rxtx:") && aa.length()>"rxtx:".length() ) {
					usaRXTX=aa.substring(aa.indexOf(":")+1).equalsIgnoreCase("on");
				}
			}

		agentLog("Usamos puerto serie: "+puertoSerie);
		if(usaRXTX) {
			//le pasamos el objeto IO
			manLMS=new ManejaLMS221(puertoSerie,new ManejaTelegramasIO());
		} else
			manLMS=new ManejaLMS221(puertoSerie);

		//Tratamos de pasar al modo 25
		try { manLMS.CambiaAModo25(); } catch (LMSException e) {
			agentLog("No fue posible pasar a modo 25");
		}

		//Configuración terminada
		agentLog("Configuracion terminada. Nos registramos");

		//Nos registramos en el DF
		DFAgentDescription dfd = new DFAgentDescription();
		dfd.setName(getAID());
		ServiceDescription sd = new ServiceDescription();
		sd.setType("comunica-LMS");
		sd.setName("comunicador-LMS");
		dfd.addServices(sd);
		try {
			DFService.register(this, dfd);
		}
		catch (FIPAException fe) {
			fe.printStackTrace();
		}
		UsandoLMS=false; //LMS disponible
		addBehaviour(new AtiendeQuerysBarrido(this));
		addBehaviour(new AtiendeQuerysConfiguracion(this));
		addBehaviour(new AtiendeQuerysPideZona(this));
		addBehaviour(new AtiendeQuerysDistancia(this));
	}
	    		
	/**
	 *  Teminación del agente.
	 *  Se desregistra el agente del DF y se cierra el GUI si existe.
	 */
	protected void takeDown() {
		// Deregister from the yellow pages
		try {
			DFService.deregister(this);
		}
		catch (FIPAException fe) {
			fe.printStackTrace();
		}

		//Cerramos el puerto
		manLMS.cierraPuerto();
		
		// Printout a dismissal message
		agentLog("Agente LMS  terminado.");
	}
		


	/**
	 * Clase privada que atiende las peticiones de barridos.
	 * 
	 * @author alberto
	 *
	 */
	private class AtiendeQuerysBarrido extends SimpleAchieveREResponder {
		private static final long serialVersionUID = 1L;
		/**
		 * Patrón de la expresión regular que deben cumplir los contenidos de los mensajes recibidos
		 */
		private Pattern patronContenido;
		
		private static final String cadenaContenidoPromedios= ":promedios";
		private static final String cadenaContenidoAnguloinicial= ":angulo-inicial";
		private static final String cadenaContenidoAngulofinal= ":angulo-final";
		
		/**
		 * Prepara el template de recepción para aceptar <i>QUERY_REF</i>.
		 * 
		 * @param a agente en que se ejecuta.
		 */
		AtiendeQuerysBarrido (Agent a) {
			super(a
					,MessageTemplate.and(
							MessageTemplate.and(
									SimpleAchieveREResponder.createMessageTemplate(FIPANames.InteractionProtocol.FIPA_QUERY)
									,MessageTemplate.MatchPerformative(ACLMessage.QUERY_REF)
							)
							,MessageTemplate.MatchOntology("Barrido")
					)
			);
		}
		
		/**
		 * Mandamos el Agree, ya que los resultados pueden tardar.
		 * Vamos a estudiar el contenido para ver que se nos pide.
		 */
		@Override
		protected ACLMessage prepareResponse(ACLMessage request)
				throws NotUnderstoodException, RefuseException {
			// Creamos mensaje de respueta y ponemos el AGREE.
			ACLMessage respuesta=request.createReply();
			//en principio no entendemos
			respuesta.setPerformative(ACLMessage.NOT_UNDERSTOOD);
			
			//valores por defecto para los 3 parámetros
			short numPromedios=1
				, anguloInicial=0
				, anguloFinal=180
				;
			//Pasamos a interpretar el contenido
			if(patronContenido==null) {
				patronContenido=Pattern.compile("\\(barrido(( +"+ cadenaContenidoPromedios
						+" +\\d+)|( +"+ cadenaContenidoAnguloinicial 
						+" +\\d+)|( +"+ cadenaContenidoAngulofinal
						+" +\\d+))* *\\)");
				agentLog("Patron inicializado a: "+patronContenido.toString());
			}
			
			String contenido=request.getContent().trim();
			if(!patronContenido.matcher(contenido).matches()) {
				agentLog("No se entiende el contenido "+contenido);
				return respuesta;
			}
		
			
			StringTokenizer st=new StringTokenizer(contenido.substring(1, contenido.length()-1));
			st.nextToken(); //quitamos el barrido
			while(st.hasMoreTokens()) {
				String slot=st.nextToken();
				if(slot.equals(cadenaContenidoPromedios))
					numPromedios=Short.valueOf(st.nextToken());
				if(slot.equals(cadenaContenidoAnguloinicial))
					anguloInicial=Short.valueOf(st.nextToken());
				if(slot.equals(cadenaContenidoAngulofinal))
					anguloFinal=Short.valueOf(st.nextToken());
			}
			
			if(UsandoLMS) {
				respuesta.setPerformative(ACLMessage.FAILURE);
				return respuesta;
			}
			try{
				manLMS.pideBarrido(anguloInicial,anguloFinal,numPromedios);
				respuesta.setPerformative(ACLMessage.AGREE);
				UsandoLMS=true;  //Nos faltar recibir la respuesta
				//si el LSM entendió el mensaje
				return respuesta;
			} catch (LMSException e) {
				agentLog("Problema "+e.getMessage());
				//mensaje de fallo
				respuesta.setPerformative(ACLMessage.FAILURE);
				return respuesta;				
			}
			
		}
		
		/**
		 * Se lo solicitamos al LMS y respondemos con el barrido.
		 * Metemos barrido en respuesta
		 */
		@Override
		protected ACLMessage prepareResultNotification(ACLMessage request,
				ACLMessage response) throws FailureException {
			ACLMessage respuesta=request.createReply();
			//en principio hay fallo
			respuesta.setPerformative(ACLMessage.FAILURE);
			UsandoLMS=false; //pase lo que pase terminaremos de usar el LMS

			
			
			try {
				BarridoAngular barr=manLMS.recibeBarrido();
				respuesta.setContentObject(barr);
				respuesta.setPerformative(ACLMessage.INFORM);
			} catch (LMSException e) {
				agentLog("Promblema: "+e.getMessage());
			} catch (java.io.IOException e) {
				agentLog("No se pudo meter objeto en mensaje por "+ e.getMessage());
			}
			return respuesta;
		}
		
	}  //de Atiende QuerysBarrido

	/**
	 * Clase privada que atiende los mensajes para modificar configuración.
	 * Contenido debe tener forma:
	 * {@code
	 *   (configuracion :rango-angular 180|100 
	 *   				:en-milimetros 0|1 
	 *   				:resolucion-angular 100|50|25
	 *   				:codigo-rango 0..6 )
	 * }
	 * @author alberto
	 *
	 */
	private class AtiendeQuerysConfiguracion extends SimpleAchieveREResponder {
		private static final long serialVersionUID = 1L;
		/**
		 * Patrón de la expresión regular que deben cumplir los contenidos de los mensajes recibidos
		 */
		private Pattern patronContenido;
		
		private static final String cadenaContenidoRangoAng= ":rango-angular";
		private static final String cadenaContenidoEnMilimetros= ":en-milimetros";
		private static final String cadenaContenidoResAng= ":resolucion-angular";
		private static final String cadenaContenidoCodRan= ":codigo-rango";
		
		private short rangoAngular;
		private boolean cambiaRangoAngular; 
		private short resAngular;
		private boolean cambiaResAngular;
		private boolean enMilimetros;
		private boolean cambiaEnMilimetros;
		private byte codigoRango;
		private boolean cambiaCodigoRango;
		

		/**
		 * Al resetear inicializamos la variables al valor supuesto
		 */
		@Override
		public void reset() {
			super.reset();
			
		}

		/**
		 * Prepara el template de recepción para aceptar <i>REQUEST</i>.
		 * 
		 * @param a agente en que se ejecuta.
		 */
		AtiendeQuerysConfiguracion (Agent a) {
			super(a
					,MessageTemplate.and(
							MessageTemplate.and(
									SimpleAchieveREResponder.createMessageTemplate(FIPANames.InteractionProtocol.FIPA_REQUEST)
									,MessageTemplate.MatchPerformative(ACLMessage.REQUEST)
							)
							,MessageTemplate.MatchOntology("Configuracion")
					)
			);
		}


		/**
		 * Mandamos el Agree, ya que los resultados pueden tardar.
		 * Vamos a estudiar el contenido para ver que se nos pide.
		 */
		@Override
		protected ACLMessage prepareResponse(ACLMessage request)
				throws NotUnderstoodException, RefuseException {
			// Creamos mensaje de respueta
			ACLMessage respuesta=request.createReply();
			//en principio no entendemos
			respuesta.setPerformative(ACLMessage.NOT_UNDERSTOOD);

			//en principio no cambiamos nada
			cambiaCodigoRango=false;
			cambiaEnMilimetros=false;
			cambiaRangoAngular=false;
			cambiaResAngular=false;

			//Pasamos a interpretar el contenido
			if(patronContenido==null) {
				patronContenido=Pattern.compile("\\(configuracion("
						+"( +" + cadenaContenidoRangoAng + " +\\d+)"
						+"|( +"+ cadenaContenidoEnMilimetros + " +(0|1))"
						+"|( +"+ cadenaContenidoResAng + " +\\d+)"
						+"|( +"+ cadenaContenidoCodRan +" +\\d+)"
					    +")* *\\)");
				agentLog("Patron inicializado a: "+patronContenido.toString());
			}
			
			String contenido=request.getContent().trim();
			if(!patronContenido.matcher(contenido).matches()) {
				agentLog("No se entiende el contenido "+contenido);
				return respuesta;
			}
		
			
			StringTokenizer st=new StringTokenizer(contenido.substring(1, contenido.length()-1));
			st.nextToken(); //quitamos el barrido
			while(st.hasMoreTokens()) {
				String slot=st.nextToken();
				if(slot.equals(cadenaContenidoRangoAng)) {
					rangoAngular=Short.valueOf(st.nextToken());
					cambiaRangoAngular=true;
				}
				if(slot.equals(cadenaContenidoResAng)) {
					resAngular=Short.valueOf(st.nextToken());
					cambiaResAngular=true;
				}
				if(slot.equals(cadenaContenidoCodRan)) {
					codigoRango=Byte.valueOf(st.nextToken());
					cambiaCodigoRango=true;
				}
				if(slot.equals(cadenaContenidoEnMilimetros)) {
					enMilimetros=!st.nextToken().equals("0");
					cambiaEnMilimetros=true;
				}
			}
		
			respuesta.setPerformative(ACLMessage.AGREE);
			respuesta.setContent(request.getContent()); //mismo contenido enviado
			return respuesta;
		}
		
		
		@Override
		protected ACLMessage prepareResultNotification(ACLMessage request, ACLMessage response) throws FailureException {
			ACLMessage respuesta=request.createReply();
			respuesta.setPerformative(ACLMessage.FAILURE);
			if(UsandoLMS)
				return respuesta;
			//A la vista de los parámetros elegimos el mensaje a mandar
			try {
				if(cambiaRangoAngular && cambiaResAngular)
					manLMS.setVariante(rangoAngular, resAngular);
				else {
					if (cambiaRangoAngular) manLMS.setRangoAngular(rangoAngular);
					if (cambiaResAngular) manLMS.setResolucionAngular(resAngular);
				}
				if(cambiaEnMilimetros && cambiaCodigoRango)
					manLMS.setCodigo(enMilimetros, codigoRango);
				else {
					if (cambiaEnMilimetros) manLMS.setEnMilimetros(enMilimetros);
					if (cambiaCodigoRango) manLMS.setCodigoRango(codigoRango);
				}
				respuesta.setPerformative(ACLMessage.INFORM);
				respuesta.setContent("(configuracion "
						+ " " + cadenaContenidoRangoAng + " " + manLMS.getRangoAngular()
						+ " " + cadenaContenidoEnMilimetros + (manLMS.isEnMilimetros()?" 1":" 0")
						+ " " + cadenaContenidoResAng + " "+ manLMS.getResolucionAngular()
						+ " " + cadenaContenidoCodRan + " "+ manLMS.getCodigoRango()
						+ " )"
				);
			} catch (LMSException e){
				agentLog("No fue posible hacer el cambio de configuracion:"+e.getMessage());

			}
			return respuesta;
		}		
		
	} // de la clase AtiendeQuerysConfiguracion

	/**
	 * Clase privada que atiende los mensajes para modificar configuración.
	 * Contenido debe tener forma:
	 * {@code
	 *   (configuracion :rango-angular 180|100 
	 *   				:en-milimetros 0|1 
	 *   				:resolucion-angular 100|50|25
	 *   				:codigo-rango 0..6 )
	 * }
	 * @author alberto
	 *
	 */
	private class AtiendeQuerysPideZona extends SimpleAchieveREResponder {
		private static final long serialVersionUID = 1L;
		/**
		 * Patrón de la expresión regular que deben cumplir los contenidos de los mensajes recibidos
		 */
		private Pattern patronContenido;
		private byte queZona;
		private boolean elConjunto1;
		
		private static final String cadenaContenidoConjunto= ":conjunto";
		private static final String cadenaContenidoZona= ":zona";
		

		/**
		 * Prepara el template de recepción para aceptar <i>REQUEST</i>.
		 * 
		 * @param a agente en que se ejecuta.
		 */
		AtiendeQuerysPideZona (Agent a) {
			super(a
					,MessageTemplate.and(
							MessageTemplate.and(
									SimpleAchieveREResponder.createMessageTemplate(FIPANames.InteractionProtocol.FIPA_REQUEST)
									,MessageTemplate.MatchPerformative(ACLMessage.REQUEST)
							)
							,MessageTemplate.MatchOntology("ZonasLMS")
					)
			);
			queZona=ZonaLMS.ZONA_A;
			elConjunto1=true;
		}

		

		@Override
		public void reset() {
			super.reset();
			queZona=ZonaLMS.ZONA_A;
			elConjunto1=true;
		}



		/**
		 * Mandamos el Agree, ya que los resultados pueden tardar.
		 * Vamos a estudiar el contenido para ver que se nos pide.
		 */
		@Override
		protected ACLMessage prepareResponse(ACLMessage request)
				throws NotUnderstoodException, RefuseException {
			// Creamos mensaje de respueta
			ACLMessage respuesta=request.createReply();
			//en principio no entendemos
			respuesta.setPerformative(ACLMessage.NOT_UNDERSTOOD);
			

			//Pasamos a interpretar el contenido
			if(patronContenido==null) {
				patronContenido=Pattern.compile("\\(zona("
						+"( +" + cadenaContenidoConjunto + " +(0|1))"
						+"|( +"+ cadenaContenidoZona + " +(A|B|C))"
					    +")* *\\)");
				agentLog("Patron inicializado a: "+patronContenido.toString());
			}
			
			String contenido=request.getContent().trim();
			if(!patronContenido.matcher(contenido).matches()) {
				agentLog("No se entiende el contenido "+contenido);
				return respuesta;
			}
		
			
			StringTokenizer st=new StringTokenizer(contenido.substring(1, contenido.length()-1));
			st.nextToken(); //quitamos el barrido
			while(st.hasMoreTokens()) {
				String slot=st.nextToken();
				if(slot.equals(cadenaContenidoZona)) {
					queZona=(byte)(st.nextToken().toCharArray()[0]-'A');
				}
				if(slot.equals(cadenaContenidoConjunto))
					elConjunto1=st.nextToken().equals("1");
			}

//			try {
//				manLMS.pideZona(queZona, elConjunto1);
			respuesta.setPerformative(ACLMessage.AGREE);
			respuesta.setContent(request.getContent()); //mismo contenido enviado
//			} catch (LMSException e) {
//				agentLog("Problema :"+e.getMessage());
//			}
			return respuesta;
		}
		
		
		@Override
		protected ACLMessage prepareResultNotification(ACLMessage request, ACLMessage response) throws FailureException {
			ACLMessage respuesta=request.createReply();
			respuesta.setPerformative(ACLMessage.FAILURE);
			
					
			try {
				ZonaLMS zn=manLMS.recibeZona(queZona,elConjunto1);
				respuesta.setContentObject(zn);
				respuesta.setPerformative(ACLMessage.INFORM);
			}
			catch (LMSException e) {
				agentLog("Problema "+e.getMessage());
			}
			catch (java.io.IOException e) {
				agentLog("No se pudo meter objeto en mensaje por "+ e.getMessage());
				e.printStackTrace();
			}
			return respuesta;

		}		
		
	} // de la clase AtiendeQuerysPideZonas

	/**
	 * Clase privada que atiende las peticiones de distancia vertical.
	 * 
	 * Atiendiende mensajes de contenido: <code>(distancia)</code>
	 * Devuelve mensajes de contenido <code>(distancia :zonaA ##  :zonaB ## :zonaC ##)</code>
	 * la distancia se indica en milímetros.
	 * 
	 * La ontología es 'Distancia'
	 * 
	 * @author alberto
	 *
	 */
	private class AtiendeQuerysDistancia extends SimpleAchieveREResponder {
		private static final long serialVersionUID = 1L;
		
		/**
		 * Que ontología usamos
		 */
		private static final String Ontologia="Distancia";
		/**
		 * Patrón de la expresión regular que deben cumplir los contenidos de los mensajes recibidos
		 */
		private Pattern patronContenido;
		
		/**
		 * Prepara el template de recepción para aceptar <i>QUERY_REF</i>.
		 * 
		 * @param a agente en que se ejecuta.
		 */
		AtiendeQuerysDistancia (Agent a) {
			super(a
					,MessageTemplate.and(
							MessageTemplate.and(
									SimpleAchieveREResponder.createMessageTemplate(FIPANames.InteractionProtocol.FIPA_QUERY)
									,MessageTemplate.MatchPerformative(ACLMessage.QUERY_REF)
							)
							,MessageTemplate.MatchOntology(Ontologia)
					)
			);
		}
		
		/**
		 * Mandamos el Agree, ya que los resultados pueden tardar.
		 * Vamos a estudiar el contenido para ver que se nos pide.
		 */
		@Override
		protected ACLMessage prepareResponse(ACLMessage request)
				throws NotUnderstoodException, RefuseException {
			// Creamos mensaje de respueta y ponemos el AGREE.
			ACLMessage respuesta=request.createReply();
			//en principio no entendemos
			respuesta.setPerformative(ACLMessage.NOT_UNDERSTOOD);
			
			//Pasamos a interpretar el contenido
			if(patronContenido==null) {
				patronContenido=Pattern.compile("\\(distancia *\\)");
				agentLog("Patron inicializado a: "+patronContenido.toString());
			}
			
			String contenido=request.getContent().trim();
			if(!patronContenido.matcher(contenido).matches()) {
				agentLog("No se entiende el contenido "+contenido);
				return respuesta;
			}
		
		    
			if(UsandoLMS) {
				respuesta.setPerformative(ACLMessage.FAILURE);
				return respuesta;
			}
						
			try { 
				manLMS.solicitaDistancia();
				UsandoLMS=true;  //Nos faltar recibir la respuesta
				//si el LSM entendió el mensaje
				respuesta.setPerformative(ACLMessage.AGREE);
			} catch (LMSException e) {
				agentLog("Problema: "+e.getMessage());
				respuesta.setPerformative(ACLMessage.FAILURE);				
			}
			return respuesta;
		}
		
		/**
		 * Se lo solicitamos al LMS y respondemos con el barrido.
		 * Metemos barrido en respuesta
		 */
		@Override
		protected ACLMessage prepareResultNotification(ACLMessage request,
				ACLMessage response) throws FailureException {
			ACLMessage respuesta=request.createReply();
			//en principio hay fallo
			respuesta.setPerformative(ACLMessage.FAILURE);
			UsandoLMS=false; //pase lo que pase terminaremos de usar el LMS
			
			try {
				double[] distancias=manLMS.recibeDistancia();
				//tenemos los datos, pasamos a informar
				respuesta.setPerformative(ACLMessage.INFORM);
				
				respuesta.setContent("(distancia"
						+" :zonaA "+String.format("%6.0f", distancias[0])
						+" :zonaB "+String.format("%6.0f",distancias[1])
	 					+" :zonaC "+String.format("%6.0f",distancias[2])
	 					                      +")"
				);
			} catch (LMSException e) {
				agentLog("Problema :"+e.getMessage());
			}

			return respuesta;
		}
		
	}  //de Atiende QuerysBarrido

}
