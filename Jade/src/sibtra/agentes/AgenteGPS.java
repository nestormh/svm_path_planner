package sibtra.agentes;


import sibtra.gps.GPSConnection;
import sibtra.gps.GPSData;
import sibtra.gps.SerialConnectionException;
import sibtra.gps.SimulaGps;
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
import jade.proto.SimpleAchieveREInitiator;
import jade.proto.SimpleAchieveREResponder;

/**
 * Agente encardado de la gestión a bajo nivel del GPS
 * @author alberto, nestor
 *
 */
public class AgenteGPS extends Agent {
	
	/**
	 * Objeto para la comunicación con el GPS
	 */
	protected GPSConnection gpsConn;

	/**
	 * Nos indica si estamos usando GPS simulado
	 */
	private boolean simula;
	
	/**
	 * Para activar o desactivar los logs
	 */
	private boolean logs;

	/**
	 * Metodo para emitir los mensajes de Log
	 */
	protected void agentLog(String m) {
		if(logs)
			System.out.println(getAID().getName()+
					": "+System.currentTimeMillis()+": "+m);
	}
	
	
	protected void setup() {
		agentLog("Se arranca el agenteGPS ");
			
		//Puerto serie por defecto
		String puertoSerie="/dev/ttyUSB0";
		logs=true;
		simula=false;
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
				if(aa.equals("simula:") || aa.equals("simula:on") ) {
					simula=true;
				}
			}

		agentLog("Usamos puerto serie: "+puertoSerie);

		//Creamos conexión segun sea real o simulada
		if(simula) {
			SimulaGps sGps=new SimulaGps(true, "../GPS/Rutas/Ruta20080612.gps");
			gpsConn= sGps.getGps();
		} else {
			try {
			gpsConn=new GPSConnection(puertoSerie);
			} catch (SerialConnectionException e) {
				agentLog("Problema al establecer conexión serial"+e.getMessage());
			}
		}

		//Configuración terminada
		agentLog("Configuracion terminada. Nos registramos");

		//Nos registramos en el DF
		DFAgentDescription dfd = new DFAgentDescription();
		dfd.setName(getAID());
		ServiceDescription sd = new ServiceDescription();
		sd.setType("comunica-GPS");
		sd.setName("comunicador-GPS");
		dfd.addServices(sd);
		try {
			DFService.register(this, dfd);
		}
		catch (FIPAException fe) {
			fe.printStackTrace();
		}
		addBehaviour(new AtiendeQuerysDatos(this));
	}
	    		
    //	==================================================================================================
	// CLASES PRIVADAS PARA LOS COMPORTAMIENTOS
    //	==================================================================================================

	protected class AtiendeQuerysDatos extends SimpleAchieveREResponder {
		
		/**
		 * 
		 */
		public AtiendeQuerysDatos(Agent a) {
			super(a
					,MessageTemplate.and(
							MessageTemplate.and(
									SimpleAchieveREResponder.createMessageTemplate(FIPANames.InteractionProtocol.FIPA_QUERY)
									,MessageTemplate.MatchPerformative(ACLMessage.QUERY_REF)
							)
							,MessageTemplate.MatchOntology("GPS")
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
			if(request.getContent().trim().equals("(UtimoPunto)")) {
				respuesta.setPerformative(ACLMessage.AGREE);				
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
			
			try {
				GPSData pto=gpsConn.getPuntoActualTemporal();
				respuesta.setContentObject(pto);
				respuesta.setPerformative(ACLMessage.INFORM);
			} catch (java.io.IOException e) {
				agentLog("No se pudo meter objeto en mensaje por "+ e.getMessage());
			}
			return respuesta;
		}
	}  // fin de AtiendeQuerysDatos

}
